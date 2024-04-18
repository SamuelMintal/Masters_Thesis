# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import time
import argparse

import pandas as pd
import torch
from nas_201_api import NASBench201API as API

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net

def get_accs_for_index(i: int, dataset_str: str) -> list[float]:
        """
        Returns 2 lists of averaged out accuracies during whole training of the architecture specified by index
        """       
        EPOCHS_COUNT = 200

        def get_val(info: dict, key: str):
            return info[key] if key in info else None

        train_accs = []
        val_accs = []

        for i_epoch in range(0, EPOCHS_COUNT): 
            info = api.get_more_info(i, dataset_str, i_epoch, hp='200', is_random=False)
            
            train_accs.append(get_val(info, 'train-accuracy'))
            val_accs.append(get_val(info, 'valid-accuracy'))

        return train_accs, val_accs 

def get_num_classes(args):
    """
    Returns 10  for Cifar 10
            100 for Cifar 100
            120 for ImageNet16-120
    """
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120

def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_0-e61699.pth', type=str, help='path to API')
    parser.add_argument('--init_w_type', type=str, default='none', help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none', help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=1, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1, help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=10, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    args = parser.parse_args()

    args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    # Load NasBench201 API
    api = API(args.api_loc)
    api.verbose = False
    
    # Set torch stuff for reproducibality
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Returns pytorch DataLoader class which return random transformed images
    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_data_workers)
    
    # Here the results will be accumulated before saving into memory
    cached_res = []
    
    # Create dir for the output files which contains experiment data
    pf = 'cf' if 'cifar' in args.dataset else 'im'
    out_dir_name = f'nb2_{pf}{get_num_classes(args)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}'
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    # Set highest index of architecture for which to compute the results
    args.end = len(api) if args.end == 0 else args.end

    # Set time measurement variables
    start = time.time()
    last_tick = start

    # Loop over NasBench201 architectures
    for i, arch_str in enumerate(api):

        # Index start/end enforces
        if i < args.start:
            continue
        if i >= args.end:
            break 

        res = {'arch_i': i, 'arch_str': arch_str}
        
        net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args))
        net.to(args.device)

        # When init_w_type and init_b_type are None this does nothing
        init_net(net, args.init_w_type, args.init_b_type)
        
        arch_str2 = nasbench2.get_arch_str_from_model(net)
        if arch_str != arch_str2:
            print(arch_str + ' != ' + arch_str2)
            raise ValueError

        measures = predictive.find_measures(
            net, 
            train_loader, 
            (args.dataload, args.dataload_info, get_num_classes(args)),
            args.device
        )

        for zc_proxy, val in measures.items():
            res[zc_proxy] = val


        # Get accuracies during whole training process which are averaged out over all runs of a architecture
        train_accs, val_accs = get_accs_for_index(i, 'cifar10-valid' if args.dataset=='cifar10' else args.dataset)
        res['train_accs'] = train_accs
        res['val_accs']   = val_accs
        
        cached_res.append(res)

        # Write cached results into file
        if ((i + 1) % args.write_freq == 0) or (i == len(api) - 1):
            
            # Save cached results
            df = pd.DataFrame(cached_res)
            df.to_csv(os.path.join(out_dir_name, f'{i + 1}.csv'), index=False)
            cached_res = []
            
            time_took_so_far = round(time.time() - start)
            time_per_arch_so_far = time_took_so_far / (i + 1)
            ETA_secs = time_per_arch_so_far * (len(api) - (i + 1))
            print(f'Last {args.write_freq} architectures took {round(time.time() - last_tick)} seconds. ETA {round(ETA_secs / 60)} minutes.')
            
            last_tick = time.time()            