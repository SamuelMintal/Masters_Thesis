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
    t_start = time()
    XGBoost_predictor.fit(train_inputs, train_targets)
    print(f'Fitting XGBoost finished in {round(time() - t_start)} seconds.')

    return XGBoost_predictor

def predict_XGBoost(XG_Boost, predict_inputs: list[list[float]]) -> list[float]:
    return XG_Boost.predict(predict_inputs)


def train_NeuralNet(train_inputs: list[list[float]], train_targets: list[float], WIDTH=128, LEN=5, overfitting_plot_path: str = None, N_EPOCHS: int | None = None) -> tf.keras.Model:
    """
    Trains neural network on supplied data

    Parameters
    ----------
    overfitting_plot_path : str | None
        If string then plots of training progress are saved together with validation loss. For this reason
        It automatically lets 10% of training data for validation purposes thus reduces train set size. If it 
        is None then no plotting is performed and thus all the train set is used for training and no validation data are used

    N_EPOCHS : int | None, default=None
        If None then is set to default 2000.
    """
    if N_EPOCHS is None:
        N_EPOCHS = 2000
    
    INPUT_LENGTH = len(train_inputs[0])
    
    inputs = tf.keras.layers.Input(shape=(INPUT_LENGTH,))
    
    x = inputs
    for _ in range(LEN):
        x = tf.keras.layers.Dense(units=WIDTH, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.MeanSquaredError()
    )

    t_start = time()
    history = model.fit(
        train_inputs,
        train_targets,
        batch_size=32,
        epochs=N_EPOCHS,
        validation_split=(0.1 if overfitting_plot_path is not None else 0),
        verbose=0
    )
    fitting_time = round(time() - t_start)
    print(f'Fitting Neural Network finished in {fitting_time} seconds.')

    if overfitting_plot_path is not None:
        fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))

        for metric in ['loss', 'val_loss']:
            ax.plot(np.arange(N_EPOCHS), history.history[metric], label=metric)

        ax.set_xlabel('Epochs trained')
        ax.set_ylabel('MSE loss value')
        ax.set_yscale('log')

        ax.set_title(f'Training progress with |train_set| = {len(train_inputs)}. Took {fitting_time} seconds.')
        ax.legend()

        os.makedirs(overfitting_plot_path, exist_ok=True)
        fig.savefig(os.path.join(overfitting_plot_path, f'training_progress_with_train_set_{len(train_inputs)}_width_{WIDTH}_len_{LEN}.png'), dpi=300)

    return model

def predict_NeuralNet(neural_net: tf.keras.Model, predict_inputs: list[list[float]]) -> list[float]:
    return neural_net.predict(predict_inputs, verbose=0)



def plot_Spearman_against_train_set_size(N_TEST_ARCHS: int, N_TRAIN_ARCHS_LIST: list[int], plots_path: str, use_None_training_epochs: bool = False):
    """
    Creates plot of Spearman against train set size as in model choice subchapter

    Parameters
    ----------
    N_TEST_ARCHS : int
        Number of architectures which will be used for testing trained predictor and
        thus calculating its Spearman's rank correlation coeficient.
    
    N_TRAIN_ARCHS_LIST : list[int]
        List of amount of architectures which will be used for training. Are disjoint with N_TEST_ARCHS

    use_None_training_epochs: bool = False
        If set to true keep defualt amount of epochs to (train set to 2000) and thus the plotting of their vlaidation learning curves occurs ~ splitting the train data into train and validation.
        Else if False found best epochs counts are used and no plotting and splitting to validation data is performed
    """
    N_TRAIN_ARCHS = max(N_TRAIN_ARCHS_LIST)

    # Initilize lists for collecting required data for plotting
    # X axis is given by N_TRAIN_ARCHS_LIST
    extrapolated_train_targets_spearman    = [] # Spearman rank corr coeff of Extrapolations, giving theoretical maximum
    XGBoost_onehot___test_set_measured_spearman   = [] # Spearman rank corr coeff of predictions of XGBoost on the test set using ONEHOT encoding
    XGBoost_std___test_set_measured_spearman      = [] # Spearman rank corr coeff of predictions of XGBoost on the test set using STANDART encoding
    NeuralNet_Small___test_set_measured_spearman = [] # Spearman rank corr coeff of predictions of Small Neural Network on the test set
    NeuralNet_Big___test_set_measured_spearman   = [] # Spearman rank corr coeff of predictions of  Big  Neural Network on the test set

    if use_None_training_epochs:
        N_Epochs_Small_NeuralNet = [None for _ in range(len(N_TRAIN_ARCHS_LIST))]
        N_Epochs_Big_NeuralNet = [None for _ in range(len(N_TRAIN_ARCHS_LIST))]
    else:
        #########train set size###  50,  100, 200, 300, 400, 500, 600
        N_Epochs_Small_NeuralNet = [300, 600, 700, 700, 700, 700, 700]
        N_Epochs_Big_NeuralNet   = [250, 600, 600, 700, 800, 900, 1000]




    # Define data getter used for experiments with ML models
    data_getter_onehot = get_optimal_data_getter(OneHotOperation_Encoder())
    data_getter_std = get_optimal_data_getter(StandartArchitecture_Encoder())
    
    ### Now lets define architectures to use together with their data
    arch_i_sampler = ArchiSampler(data_getter_onehot.get_amount_of_architectures())

    # Get testing architectures first, so they are always the same regardless of N_TRAIN_ARCHS used
    # On these we will be testing Spearman's rank corr coef
    test_archs_i = [arch_i_sampler.sample() for _ in range(N_TEST_ARCHS)]
    # And get their data for to be used for testing
    test_inputs_onehot, test_real_valaccs = data_getter_onehot.get_testing_data_for_arch_indices(test_archs_i)
    test_inputs_std, _ = data_getter_std.get_testing_data_for_arch_indices(test_archs_i)

    # Now get training data archs_i
    train_archs_i = [arch_i_sampler.sample() for _ in range(N_TRAIN_ARCHS)]

    # And also note real validation accuracies of train_archs_i for theroeticaly maximum Spearman calculation
    train_real_valaccs = np.array([data_getter_onehot.get_real_val_acc_of_arch(arch_i) for arch_i in train_archs_i])


    for i, N_TRAIN_ARCHS in enumerate(N_TRAIN_ARCHS_LIST):
        # transform train_archs_i into training inputs and targets (LC Extrapolation happens here)
        print(f'Extrapolating |train_set| = {N_TRAIN_ARCHS}')
        train_inputs_onehot, train_targets = data_getter_onehot.get_training_data_by_arch_indices(train_archs_i[:N_TRAIN_ARCHS])
        train_inputs_std = np.array([data_getter_std.get_prediction_features_by_arch_index(train_arch_i) for train_arch_i in train_archs_i[:N_TRAIN_ARCHS]])


        # Calculate maximal theoreticaly achievable Spearman
        extrapolated_train_targets_spearman.append(
            stats.spearmanr(train_targets, train_real_valaccs[:N_TRAIN_ARCHS]).statistic
        )
        ###
        ### Get the Spearman of XGBoosts models
        ###
        # Starting with one hot encoding using one
        print(f'Starting to train onehot XGBoost with |train_set| = {N_TRAIN_ARCHS}')
        XGBoost_predictor = train_XGBoost(train_inputs_onehot, train_targets)

        # Get Spearman of its predictions on the test set
        test_predictions = predict_XGBoost(XGBoost_predictor, test_inputs_onehot)

        XGBoost_onehot___test_set_measured_spearman.append(
            # Note that we always use entire testing set
            stats.spearmanr(test_real_valaccs, test_predictions).statistic
        )  

        # Now moving onto the standart encoding usin one
        print(f'Starting to train standart enc XGBoost with |train_set| = {N_TRAIN_ARCHS}')
        XGBoost_predictor = train_XGBoost(train_inputs_std, train_targets)

        # Get Spearman of its predictions on the test set
        test_predictions = predict_XGBoost(XGBoost_predictor, test_inputs_std)

        XGBoost_std___test_set_measured_spearman.append(
            # Note that we always use entire testing set
            stats.spearmanr(test_real_valaccs, test_predictions).statistic
        )



        ###
        ### Now do the same for the both neural network based predictor
        ###
        # Starting with small one
        print(f'Starting to train small Neural Network with |train_set| = {N_TRAIN_ARCHS}')
        small_NN_model = train_NeuralNet(
            train_inputs_onehot, 
            train_targets, 
            WIDTH=128, 
            LEN=5, 
            overfitting_plot_path=(os.path.join(plots_path, 'small_NN') if use_None_training_epochs else None), 
            N_EPOCHS=N_Epochs_Small_NeuralNet[i]
        )
        test_predictions = predict_NeuralNet(small_NN_model, test_inputs_onehot)

        NeuralNet_Small___test_set_measured_spearman.append(
            # Note that we always use entire testing set
            stats.spearmanr(test_real_valaccs, test_predictions).statistic
        )
        # And Big one
        print(f'Starting to train big Neural Network with |train_set| = {N_TRAIN_ARCHS}')
        big_NN_model = train_NeuralNet(
            train_inputs_onehot, 
            train_targets, 
            WIDTH=128*2, 
            LEN=5*2, 
            overfitting_plot_path=(os.path.join(plots_path, 'big_NN') if use_None_training_epochs else None), 
            N_EPOCHS=N_Epochs_Big_NeuralNet[i]
        )
        test_predictions = predict_NeuralNet(big_NN_model, test_inputs_onehot)

        NeuralNet_Big___test_set_measured_spearman.append(
            # Note that we always use entire testing set
            stats.spearmanr(test_real_valaccs, test_predictions).statistic
        )
        

    # And now plot the results
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))

    ax.plot(N_TRAIN_ARCHS_LIST, extrapolated_train_targets_spearman, 'o--', label='Extrapolated targets of train set')
    ax.plot(N_TRAIN_ARCHS_LIST, XGBoost_onehot___test_set_measured_spearman, 'o--', label='XGBoost\'s predictions on test set (One hot encoding)')
    ax.plot(N_TRAIN_ARCHS_LIST, XGBoost_std___test_set_measured_spearman, 'o--', label='XGBoost\'s predictions on test set (Standart encoding)')
    ax.plot(N_TRAIN_ARCHS_LIST, NeuralNet_Small___test_set_measured_spearman, 'o--', label='Small Neural Network\'s predictions on test set')
    ax.plot(N_TRAIN_ARCHS_LIST, NeuralNet_Big___test_set_measured_spearman, 'o--', label='Big Neural Network\'s predictions on test set')

    ax.set_ylabel('Spearman rank correlation coefficient')
    ax.set_xlabel('Amount of architectures in the train set')
    ax.legend()

    ax.set_title(f'Dependence of the rank correlation on the size of the train set')

    os.makedirs(plots_path, exist_ok=True)
    fig.savefig(os.path.join(plots_path, f'Spearman_rank_corr_coef_vs_train_set_size.png'), dpi=300)
    
    ##########################################
    ##########################################
    ##########################################
    print('extrapolated_train_targets_spearman')
    print(extrapolated_train_targets_spearman)

    print('XGBoost_std___test_set_measured_spearman')
    print(XGBoost_std___test_set_measured_spearman)    

    print('XGBoost_onehot___test_set_measured_spearman')
    print(XGBoost_onehot___test_set_measured_spearman)
    
    print('NeuralNet_Small___test_set_measured_spearman')
    print(NeuralNet_Small___test_set_measured_spearman)
    
    print('NeuralNet_Big___test_set_measured_spearman')
    print(NeuralNet_Big___test_set_measured_spearman)
    



        



def main(path_to_fig_dir: str):
    
    plot_Spearman_against_train_set_size(
        1000, 
        [50, 100, 200, 300, 400, 500, 600],
        os.path.join(path_to_fig_dir, 'ML_models_Spearman_progression'),
        use_None_training_epochs=False # If False uses discovered training epochs for NNs, if True than runs 2k epochs in order to figure out the not-overfitting ones
    )

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    main('plot_thesis_figures')