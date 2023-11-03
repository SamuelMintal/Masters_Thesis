from ast import literal_eval
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(CLEAN_DATA_PATH: str = os.path.join('___my_code', '_clean_all_merged.csv')) -> dict[int,dict]:
    """
    Loads data and returns them in dictionary form.
    In this dictionary keys are architecture indices as in NasBench201.
    """
    CONVERT_TO_LIST_COLUMNS = ['train_accs', 'val_accs']

    # Load data
    data_df = pd.read_csv(CLEAN_DATA_PATH, index_col='arch_i')
    for conv_to_list_col in CONVERT_TO_LIST_COLUMNS:
        data_df[conv_to_list_col] = data_df[conv_to_list_col].apply(literal_eval)

    data_dict = data_df.to_dict('index')

    return data_dict

class ArchiSampler:
    """
    Stores archi_i indices in cyclic buffer. order of arch_i is randomized (using np seed)
    """
    def __init__(self, amount_of_archs: int) -> None:
        self.shuffled_architecture_indices = np.random.shuffle(np.arange(amount_of_archs))
        self.cur_idx = 0

    def sample(self) -> int:
        """
        Returns another arch_i
        """
        # get arch_i for returning
        res_arch_i = self.shuffled_architecture_indices[self.cur_idx]

        # Advance index to buffer
        self.cur_idx = (self.cur_idx + 1) % len(self.shuffled_architecture_indices)

        return res_arch_i