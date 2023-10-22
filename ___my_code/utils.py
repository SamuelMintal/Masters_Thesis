from ast import literal_eval
import os

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