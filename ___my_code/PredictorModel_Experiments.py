import os
import random
from functools import partial
from typing import Callable
from time import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from scipy import stats


from utils import ArchiSampler
from ArchitectureEncoder import *
from PredictorDataGetter import DataGetter
from target_extrapolations import prefit_avg_initialization_extrapolation
from LearningCurveExtrapolator import LearningCurveExtrapolator, LogShiftScale_Extrapolator, VaporPressure_Extrapolator, Pow3_Extrapolator


def get_optimal_prefit_avg_partialed_func() -> Callable:
    """
    returns prefit_avg_initialization_extrapolation function with hyperparameters
    being fixed to the optimal values as found by Extrapolations Experiments only
    requiring 2 inputs:
        1st being the list of lcs (uses only first 20 epochs as done in thesis)
        2nd being the average lc (again uses only first 20 epochs)
    """

    def get_top_three_extrapolators(lr: float) -> list[LearningCurveExtrapolator]:
            return [
                LogShiftScale_Extrapolator(lr=lr),
                VaporPressure_Extrapolator(lr=lr), 
                Pow3_Extrapolator(lr=lr), 
            ]
    
    return partial(
        prefit_avg_initialization_extrapolation, 
        TIME_PER_EXTRAPOLATOR_PER_LC=5, 
        get_extrapolators=get_top_three_extrapolators, 
        lrs=(0.2, 0.05), 
        prefit_time_split=0.05,
        EPOCHS_PARTIAL_LC=20
    )


def get_optimal_data_getter(arch_encoder: ArchitectureEncoder) -> DataGetter:
    """
    Returns optimal data getter (In sense of initialization strategy) using specified `arch_encoder`.
    """

    return DataGetter(
        arch_encoder,
        ['grad_norm', 'snip', 'grasp', 'jacob_cov', 'synflow_bn'],
        get_optimal_prefit_avg_partialed_func()
    )

def train_XGBoost(train_inputs: list[list[float]], train_targets: list[float]):
    """
    Trains and returns XGBoost based predictor on supplied data
    """
    XGBoost_predictor = XGBRegressor()
    print('Fitting XGBoost started')
    t_start = time()
    XGBoost_predictor.fit(train_inputs, train_targets)
    print(f'Fitting XGBoost finished in {round(time() - t_start)} seconds.')

    return XGBoost_predictor

def predict_XGBoost(XG_Boost, predict_inputs: list[list[float]]) -> list[float]:
    return XG_Boost.predict(predict_inputs)


def train_n_predict():
    pass




def plot_Spearman_against_train_set_size(N_TEST_ARCHS: int, N_TRAIN_ARCHS_LIST: list[int], plots_path: str):
    """
    Creates plot of Spearman against train set size as in model choice subchapter

    Parameters
    ----------
    N_TEST_ARCHS : int
        Number of architectures which will be used for testing trained predictor and
        thus calculating its Spearman's rank correlation coeficient.
    
    N_TRAIN_ARCHS_LIST : list[int]
        List of amount of architectures which will be used for training. Are disjoint with N_TEST_ARCHS
    """
    N_TRAIN_ARCHS = max(N_TRAIN_ARCHS_LIST)

    # Define data getter used for experiments with ML models
    data_getter_onehot = get_optimal_data_getter(OneHotOperation_Encoder())
    

    ### Now lets define architectures to use together with their data
    arch_i_sampler = ArchiSampler(data_getter_onehot.get_amount_of_architectures())

    # Get testing architectures first, so they are always the same regardless of N_TRAIN_ARCHS used
    # On these we will be testing Spearman's rank corr coef
    test_archs_i = [arch_i_sampler.sample() for _ in range(N_TEST_ARCHS)]
    # And get their data for to be used for testing
    test_inputs, test_real_valaccs = data_getter_onehot.get_testing_data_for_arch_indices(test_archs_i)

    # Now get training data (includes extrapolation of the targets defined in DataGetter)
    train_archs_i = [arch_i_sampler.sample() for _ in range(N_TRAIN_ARCHS)]
    # transform train_archs_i into training inputs and targets (LC Extrapolation happens here)
    train_inputs, train_targets = data_getter_onehot.get_training_data_by_arch_indices(train_archs_i)
    # And also note real validation accuracies of train_archs_i for theroeticaly maximum Spearman calculation
    train_real_valaccs = [data_getter_onehot.get_real_val_acc_of_arch(arch_i) for arch_i in train_archs_i]
    
    # Initilize lists for collecting required data for plotting
    # X axis is given by N_TRAIN_ARCHS_LIST
    extrapolated_train_targets_spearman    = [] # Spearman rank corr coeff of Extrapolations, giving theoretical maximum
    XGBoost___test_set_measured_spearman   = [] # Spearman rank corr coeff of predictions of XGBoost on the test set
    NeuralNet___test_set_measured_spearman = [] # Spearman rank corr coeff of predictions of Neural Network on the test set

    for N_TRAIN_ARCHS in N_TRAIN_ARCHS_LIST:
        # Limit training data to N_TRAIN_ARCHS samples
        cur_train_inputs, cur_train_targets = train_inputs[:N_TRAIN_ARCHS], train_targets[:N_TRAIN_ARCHS]

        # Calculate maximal theoreticaly achievable Spearman
        extrapolated_train_targets_spearman.append(
            stats.spearmanr(cur_train_targets, train_real_valaccs[:N_TRAIN_ARCHS]).statistic
        )

        print(f'Starting to train XGBoost with |train_set| = {N_TRAIN_ARCHS}')
        XGBoost_predictor = train_XGBoost(cur_train_inputs, cur_train_targets)

        # Get Spearman of its predictions on the test set
        test_predictions = predict_XGBoost(XGBoost_predictor, test_inputs)

        XGBoost___test_set_measured_spearman.append(
            # Note that we always use entire testing set
            stats.spearmanr(test_real_valaccs, test_predictions).statistic
        )

        # Now do the same for the neural network based predictor

    
    # And now plot the results
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))

    ax.plot(N_TRAIN_ARCHS_LIST, extrapolated_train_targets_spearman, 'o--', label='Extrapolated targets of train set')
    ax.plot(N_TRAIN_ARCHS_LIST, XGBoost___test_set_measured_spearman, 'o--', label='XGBoost\'s predictions on test set')
    ax.plot(N_TRAIN_ARCHS_LIST, NeuralNet___test_set_measured_spearman, 'o--', label='Neural Network\'s predictions on test set')

    ax.set_ylabel('Spearman rank correlation coefficient')
    ax.set_xlabel('Amount of architectures in the train set')
    ax.legend()

    ax.set_title(f'Dependence of the rank correlation on the size of the test set')

    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, f'Spearman_rank_corr_coef_vs_train_set_size.png'), dpi=300)




        





def main(path_to_fig_dir: str):
    plot_Spearman_against_train_set_size(
        1000, 
        [50, 100, 200, 300, 400],
        os.path.join(path_to_fig_dir, 'ML_models_Spearman_progression')
    )

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    main('plot_thesis_figures')