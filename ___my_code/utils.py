from ast import literal_eval
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PopulationElement:
    """
    Class representing member of population for AE algo
    """
    def __init__(self, arch_i: int, predicted_val_acc: float, real_val_acc: float) -> None:
        self.arch_i = arch_i
        self.predicted_val_acc = predicted_val_acc
        self.real_val_acc = real_val_acc


# TODO: get those values
class Constants:
    AVG_TIME_PER_ARCH_TRAINING_EPOCH = 0


def calc_spearman_rank_correlation_coef(data_real: dict[str, float], data_predicted: dict[str, float], also_plot: bool = True) -> float:
    """
    Calculates Spearman's rank correlation coefficient for given data.

    Parameters
    ----------
    data_[real,predicted] : dict[str, float]
        Dictionaries with same keys, where values are values based on which 
        we want to calculate ranking of the keys

    also_plot : bool, default=True
        Whether to also plot the values from which the correlation coefficient
        is to be calculated
    """
    def sort_dict_based_on_valaccs(dict: dict[str, float]) -> list[tuple[str, float]]:
        return sorted(dict.items(), key=lambda x: x[1])
    
    # Sort architectures based on theyr validation accuracies
    data_real_valaccs_sorted = sort_dict_based_on_valaccs(data_real)
    data_predicted_valaccs_sorted = sort_dict_based_on_valaccs(data_predicted)

    def change_valaccs_for_ranks(data: list[tuple[str, float]]) -> list[tuple[str, float]]:
        ranks = np.arange(len(data))
        data_labels = [k for (k, v) in data]

        return [(k, rank) for (k, rank) in zip(data_labels, ranks)]
    
    # Now put the rank to the architectures
    data_real_with_ranks = change_valaccs_for_ranks(data_real_valaccs_sorted)
    data_predicted_with_ranks = change_valaccs_for_ranks(data_predicted_valaccs_sorted)

    def sort_tuples_by_archs(data: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return sorted(data)
    
    # Now sort the data so the matching architectures are on matching indices
    data_real_sorted_archs = sort_tuples_by_archs(data_real_with_ranks)
    data_predicted_sorted_archs = sort_tuples_by_archs(data_predicted_with_ranks)

    # Now we just extract ranks of these architectures 
    ranks_real = np.array([r for (k, r) in data_real_sorted_archs])
    ranks_predicted = np.array([r for (k, r) in data_predicted_sorted_archs])

    ranks_detlas = ranks_real - ranks_predicted
    ranks_deltas_squared = ranks_detlas * ranks_detlas
    n = len(ranks_real)

    sp_rank_coef = 1 - (6 * np.sum(ranks_deltas_squared)) / (n * ((n ** 2) - 1))

    if also_plot:
        # Sort them by architecture so we can plot them (they are matched by architecture by index)
        data_real_plotting = sort_tuples_by_archs(list(data_real.items()))
        data_predicted_plotting = sort_tuples_by_archs(list(data_predicted.items()))

        # Plot comaprison of real and predicted validation accuracies
        real_valaccs = [valacc for (arch, valacc) in data_real_plotting]
        pred_valaccs = [valacc for (arch, valacc) in data_predicted_plotting]
        plt.scatter(real_valaccs, pred_valaccs)
        
        # Plot y=x line
        min_coord = min([min(real_valaccs), min(pred_valaccs)])
        max_coord = max([max(real_valaccs), max(pred_valaccs)])
        plt.plot([min_coord, max_coord], [min_coord, max_coord], color='black', label='y=x')

        plt.xlabel('Real validation accuracy of architectures [%]')
        plt.ylabel('Predicted validation accuracy of architectures [%]')
        plt.title(f'Real vs Predicted validation accuracy of architectures (Spearman = {round(sp_rank_coef, 2)})')
        plt.legend()
        plt.show()

    return sp_rank_coef




def plot_history(history: list[PopulationElement], POP_INIT_SIZE: int):
    """
    plots run of the AE algo given it's history and initial size of the population.
    """

    def get_best_val_acc_so_far(val_accs: list[float], POP_INIT_SIZE: int) -> list[float]:
        res = []
        for i in range(len(val_accs)):
            res.append(max(val_accs[:i + 1]))

        # Aggregate initial population val accs into 1 
        return [max(res[:POP_INIT_SIZE])] + res[POP_INIT_SIZE:] 


    predicted_val_accs = [pop_elem.predicted_val_acc for pop_elem in history]
    real_val_accs      = [pop_elem.real_val_acc      for pop_elem in history]

    best_predicted_val_acc_so_far = get_best_val_acc_so_far(predicted_val_accs, POP_INIT_SIZE)
    best_real_val_acc_so_far      = get_best_val_acc_so_far(real_val_accs, POP_INIT_SIZE)
    
    evol_loops = [i for i in range(len(best_real_val_acc_so_far))]

    plt.plot(evol_loops, best_real_val_acc_so_far, label='Real val acc')
    plt.plot(evol_loops, best_predicted_val_acc_so_far, label='Predicted val acc')
    plt.legend()
    plt.show()


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
        self.shuffled_architecture_indices = np.arange(amount_of_archs)
        np.random.shuffle(self.shuffled_architecture_indices)

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