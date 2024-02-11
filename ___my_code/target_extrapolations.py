from typing import Callable

import os
import numpy
import numpy as np
import matplotlib.pyplot as plt

from LearningCurveExtrapolator import LearningCurveExtrapolator, LearningCurveExtrapolatorsEnsembler

####################################################################################################
################################## Helper Functions ################################################
####################################################################################################

def get_MAE_of_fit(train_data: list[float], predictions: list[float]) -> float:
    """
    Returns MAE between data and predictions
    """
    return np.mean(np.abs(np.array(train_data) - np.array(predictions)))

def plot_ensembler_prediction_on_data(ax: plt.Axes, lce_ensembler: LearningCurveExtrapolatorsEnsembler, whole_lc: list[float], trained_on_n_samples: int, title: str):
    """
    Plots how Ensembler and its individual extrapolators extrapolate given data onto given Axis `ax`.
    """
    epochs = np.arange(len(whole_lc)) + 1

    ax.plot(epochs, whole_lc, label='Ground truth', color='black')
    ax.plot(epochs, [lce_ensembler.predict_avg(i) for i in epochs], label='Ensemble average')

    for lce in lce_ensembler.show_extrapolators_list():
        ax.plot(epochs, [lce.predict(i) for i in epochs], label=lce.get_name())
        
    ax.axvline(trained_on_n_samples, color='black')

    ax.set_xlabel('Training epoch of architecture')
    ax.set_ylabel('Validation accuracy [%]')
    ax.set_title(title)
    
    ax.legend()

####################################################################################################
################################## Initialization approaches #######################################
####################################################################################################


def standart_initialization_extrapolation(
        input_lcs: list[list[float]], 
        input_average_lc: list[list[float]],
        TIME_PER_EXTRAPOLATOR_PER_LC: float,
        get_extrapolators: Callable,
        lr: float,
        EPOCHS_PARTIAL_LC: int = 20,
        plots_path: str = None,
        verbose: bool = True
    ) -> list[float]:
    """
    Extrapolates and returns targets of `input_lc` based on standard initialization approach.

    Parameters
    ----------
    input_lcs : list[list[float]]
        List of whole learning curves to fit

    input_average_lc : list[list[float]]
        Averaged out curve from `input_lcs` curves (Used only for plotting here)
    
    TIME_PER_EXTRAPOLATOR_PER_LC : float
        Time in seconds how long can each extrapolator fit. (20 seconds = 1 architecture's training epoch)

    get_extrapolators : Callable
        Function returning extrapolators with default values accepting learning rate parameter

    lr : float
        Learning rate to be used for fitting

    EPOCHS_PARTIAL_LC : int, default=20
        How many epochs of the input learning curves are training data

    plots_path : str, default=None
        If not None also plots graphs used in thesis and saves them into directory `plots_path`

    verbose : bool, default=True
        Whether to print progress or not
    """
    LCS_TO_FIT = len(input_lcs) 
    LC_EPOCH_COUNT = len(input_average_lc)
    val_acc_predictions = []

    # If I am supposed to be plotting initialize dir for it
    if plots_path is not None:
        MAE_list = []
        os.makedirs(plots_path, exist_ok=True)
    
    # Go trhough each input learning curve
    for i, whole_lc in enumerate(input_lcs):
        if verbose:
            print(f'Fitting LC {i + 1}/{LCS_TO_FIT}...')
        
        # Isolate training part of the current LC
        train_lc = whole_lc[:EPOCHS_PARTIAL_LC]

        # Init and fit Ensmebler
        extr_ensemble = LearningCurveExtrapolatorsEnsembler(get_extrapolators(lr=lr), verbose=verbose)
        extr_ensemble.fit_extrapolators(train_lc, time_seconds=TIME_PER_EXTRAPOLATOR_PER_LC)

        # Note its predictions
        val_acc_predictions.append(
            extr_ensemble.predict_avg(LC_EPOCH_COUNT)
        )

        # If we want to plot stuff
        if plots_path is not None:
            # Note the MAE of the fit
            mae_of_fit = get_MAE_of_fit(train_lc, [extr_ensemble.predict_avg(i + 1) for i in range(EPOCHS_PARTIAL_LC)])
            MAE_list.append(mae_of_fit)

    # Create the final plot if is supposed to
    if plots_path is not None:
        # Get the average MAE of fit
        final_avg_MAE = np.average(MAE_list)

        # Fit the Ensmebler on the average curve to have Ensembler for the plot
        avg_train_lc = input_average_lc[:EPOCHS_PARTIAL_LC]
        avg_extr_ensemble = LearningCurveExtrapolatorsEnsembler(get_extrapolators(lr=lr), verbose=verbose)
        if verbose:
            print(f'Fitting averaged LC...')
        avg_extr_ensemble.fit_extrapolators(avg_train_lc, time_seconds=TIME_PER_EXTRAPOLATOR_PER_LC)

        # Create and save the plot from the acquired data
        fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))
        plot_ensembler_prediction_on_data(ax, avg_extr_ensemble, input_average_lc, EPOCHS_PARTIAL_LC, f'For lr = {lr} we have average MAE of fit = {round(final_avg_MAE, 5)}')
        fig.savefig(os.path.join(plots_path, f'Standart_init_approach_lr_{lr}.png'))

    # Return all predicted validation accuracies of `input_lcs`
    return val_acc_predictions

def prefit_avg_initialization_extrapolation(
        input_lcs: list[list[float]], 
        input_average_lc: list[list[float]],
        TIME_PER_EXTRAPOLATOR_PER_LC: float,
        get_extrapolators: Callable,
        lrs: tuple[float, float],
        prefit_time_split: float,
        EPOCHS_PARTIAL_LC: int = 20,
        plots_path: str = None,
        verbose: bool = True
    ) -> list[float]:
    """
    Extrapolates and returns targets of `input_lc` based on standard initialization approach.

    Parameters
    ----------
    input_lcs : list[list[float]]
        List of whole learning curves to fit

    input_average_lc : list[list[float]]
        Averaged out curve from `input_lcs` curves used for prefitting
    
    TIME_PER_EXTRAPOLATOR_PER_LC : float
        Time in seconds how long can each extrapolator fit. (20 seconds = 1 architecture's training epoch)

    get_extrapolators : Callable
        Function returning extrapolators with default values accepting learning rate parameter

    lrs : tuple[float, float]
        Tuple of two learning rates to be used for fitting.
        First is the lr for fitting averaged lc and second lr is for fitting concrete lc.

    prefit_time_split : float
        Fraction [0,1] defining proportion of overall time which is spent fitting the averaged curve
        and thus defining the time left for fitting concrete lcs

    EPOCHS_PARTIAL_LC : int, default=20
        How many epochs of the input learning curves are training data

    plots_path : str, default=None
        If not None also plots graphs used in thesis and saves them into directory `plots_path`

    verbose : bool, default=True
        Whether to print progress or not
    """
    val_acc_predictions = []
    lr_prefit, lr_concrete_fit = lrs[0], lrs[1]

    LCS_TO_FIT = len(input_lcs) 
    LC_EPOCH_COUNT = len(input_average_lc)
    
    # Compute time budgets for prefitting and concrete lc fitting
    TOTAL_TIME_PER_EXTRAPOLATOR = LCS_TO_FIT * TIME_PER_EXTRAPOLATOR_PER_LC
    PREFIT_TIME_PER_EXTRAPOLATOR = TOTAL_TIME_PER_EXTRAPOLATOR * prefit_time_split
    CONCRETE_LC_TIME_PER_EXTRAPOLATOR = (TOTAL_TIME_PER_EXTRAPOLATOR - PREFIT_TIME_PER_EXTRAPOLATOR) / LCS_TO_FIT

    if plots_path is not None:
        MAE_list = []
        os.makedirs(plots_path, exist_ok=True)

    # Do the prefitting
    prefitted_extr_ensemble = LearningCurveExtrapolatorsEnsembler(get_extrapolators(lr=lr_prefit), verbose=verbose)
    prefitted_extr_ensemble.fit_extrapolators(input_average_lc[:EPOCHS_PARTIAL_LC], time_seconds=PREFIT_TIME_PER_EXTRAPOLATOR)

    # Now go trhough each input learning curve
    # Here simply copy prefitted Ensembler and fit it additionaly
    for i, whole_lc in enumerate(input_lcs):
        if verbose:
            print(f'Fitting LC {i + 1}/{LCS_TO_FIT}...')

        # Isolate training part of the current LC
        train_lc = whole_lc[:EPOCHS_PARTIAL_LC]

        # Copy the ensemble and change its learning rate
        concrete_extr_ensemble = prefitted_extr_ensemble.copy()
        concrete_extr_ensemble.change_lr(lr_concrete_fit)

        # Fit it to the train data
        concrete_extr_ensemble.fit_extrapolators(train_lc, time_seconds=CONCRETE_LC_TIME_PER_EXTRAPOLATOR)

        # Note its predictions
        val_acc_predictions.append(
            concrete_extr_ensemble.predict_avg(LC_EPOCH_COUNT)
        )

        # If we want to plot stuff
        if plots_path is not None:
            # Note the MAE of the fit
            mae_of_fit = get_MAE_of_fit(train_lc, [concrete_extr_ensemble.predict_avg(i + 1) for i in range(EPOCHS_PARTIAL_LC)])
            MAE_list.append(mae_of_fit)



    # Create the final plot if is supposed to
    if plots_path is not None:
        # Get the average MAE of fit
        final_avg_MAE = np.average(MAE_list)

        # Get the training data
        avg_train_lc = input_average_lc[:EPOCHS_PARTIAL_LC]

        # Fit the Ensmebler on the average curve to have Ensembler for the plot
        avg_extr_ensemble = prefitted_extr_ensemble.copy()
        avg_extr_ensemble.change_lr(lr_concrete_fit)
        if verbose:
            print(f'Fitting averaged LC...')
        avg_extr_ensemble.fit_extrapolators(avg_train_lc, time_seconds=CONCRETE_LC_TIME_PER_EXTRAPOLATOR)

        # Create and save the plot from the acquired data
        fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))
        plot_ensembler_prediction_on_data(ax, avg_extr_ensemble, input_average_lc, EPOCHS_PARTIAL_LC, f'For lr = ({lr_prefit}, {lr_concrete_fit}) and {round(prefit_time_split*100)}% of time spent on fitting the average curve we have average MAE of fit = {round(final_avg_MAE, 5)}')
        fig.savefig(os.path.join(plots_path, f'Prefit_average_init_approach_lr_{lr_prefit}_{lr_concrete_fit}_frac_{prefit_time_split}.png'))


    # Return all predicted validation accuracies of `input_lcs`
    return val_acc_predictions