import os

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from nas_201_api import NASBench201API

class DataGetter():
    def __init__(self, api: NASBench201API, path_to_csv_data: str = os.path.join('___my_code', '_clean_all_merged.csv'), dataset_api: str = 'cifar10-valid', EPOCHS_COUNT: int = 200) -> None:
    
        df = pd.read_csv(path_to_csv_data)#.set_index('arch_i', drop=True)
        self.data_df = df.to_dict('index')

        self.cached_valid_accs = {}

        self.api = api
        self.dataset_api = dataset_api
        self.EPOCHS_COUNT = EPOCHS_COUNT

    def get_valid_accs_for_index(self, i: int) -> list[float]:
        """
        Returns list of averaged out validation accuracies of the architecture specified by index
        """
        val_accs = []

        for i_epoch in range(0, self.EPOCHS_COUNT):
            info = self.api.get_more_info(i, self.dataset_api, i_epoch, hp='200', is_random=False)
            val_accs.append(info['valid-accuracy'])

        return val_accs

    def __getitem__(self, key) -> tuple[dict[str, float], list[float]]:
        """
        Returns tuple containing:
            1. dictionary of Data for the index-ed architecture
            2. list of averaged out validation accuracies for the index-ed architecture
        """    
        # Get the validation accuracies if they are not cached
        if key not in self.cached_valid_accs:
            self.cached_valid_accs[key] = self.get_valid_accs_for_index(key)
        
        return self.data_df[key], self.cached_valid_accs[key]



if __name__ == '__main__':
    INDEX = 0
    dg = DataGetter(api=NASBench201API(os.path.join('data', 'NAS-Bench-201-v1_0-e61699.pth')))
    data, val_accs = dg[INDEX]

    plt.plot([i for i in range(len(val_accs))], val_accs)
    plt.title(f'Validation accuracies of architecture {INDEX}')
    plt.show()
    