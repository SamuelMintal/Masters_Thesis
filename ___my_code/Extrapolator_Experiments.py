import os
import random
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LearningCurveExtrapolator import *
from utils import ArchiSampler, load_data



def get_learning_curves_data(amount_of_archs: int) -> tuple[list[list[float]], list[float]]:
    """
    Uses np.seed to base its selection.

    Returns whole learning curves of `amount_of_archs` random architectures and averaged out learning curve
    """
    data = load_data()
    assert len(data) == 15625, 'Missing architectures in _clean_all_merged.csv'
    
    arch_i_sampler = ArchiSampler(len(data))    

    whole_lcs = []
    avg_whole_lc = np.zeros(200)
    for _ in range(amount_of_archs):
        val_accs = np.array(data[arch_i_sampler.sample()]['val_accs'])

        avg_whole_lc += val_accs
        whole_lcs.append(val_accs)

    avg_whole_lc = avg_whole_lc / amount_of_archs
        

    return whole_lcs, avg_whole_lc


def get_fresh_extrapolators(lr: float) -> list[LearningCurveExtrapolator]:
        return [
            LogShiftScale_Extrapolator(lr=lr),
            VaporPressure_Extrapolator(lr=lr), 
            Pow3_Extrapolator(lr=lr), 
            LogLogLinear_Extrapolator(lr=lr),
            LogPower(lr=lr),
            Pow4_Extrapolator(lr=lr),
            MMF_Extrapolator(lr=lr),
            Exp4_Extrapolator(lr=lr),
            Janoschek_Extrapolator(lr=lr),
            Weibull_Extrapolator(lr=lr)            
        ]


def get_top_three_extrapolators(lr: float) -> list[LearningCurveExtrapolator]:
        return [
            LogShiftScale_Extrapolator(lr=lr),
            VaporPressure_Extrapolator(lr=lr), 
            Pow3_Extrapolator(lr=lr), 
        ]



###################################################
############ Parametric mdoels tests ##############
###################################################


def plot_progress_of_extrapolators(extrapolators_results: dict[str, dict], golden_whole_lc: list[float], save_name_suffix: str = '', EPOCHS_PARTIAL_LC: int = 20):
    """
    Plots progress of extrapolators fittings on curve
    """
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))
    ax.axvline(EPOCHS_PARTIAL_LC, color='black')

    epochs_overall = np.arange(len(golden_whole_lc)) + 1

    # Plot their prediction curves and ground truth
    for extr_name, extr_data in extrapolators_results.items():
        extrapolator: LearningCurveExtrapolator = extr_data['extrapolator'] 

        ax.plot(epochs_overall, [extrapolator.predict(i) for i in epochs_overall], label=extr_name + f' (Δp = {round(abs(extrapolator.predict(200) - golden_whole_lc[-1]), 2)})')

    ax.plot(epochs_overall, golden_whole_lc, label='Ground truth', color='black')   
    
    ax.set_xlabel('Training epoch of architecture')
    ax.set_ylabel('Validation accuracy [%]')

    ax.legend()
    ax.set_title('Prediction of entire learning curve') 
    plt.savefig(os.path.join('plot_thesis_figures', f'predictions_x_ground_truth_{save_name_suffix}.png'), dpi=300)


    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))

    # Plot their MAE progress
    for extr_name, extr_data in extrapolators_results.items():
        mae_progress = extr_data['mae_progress']
        at_x_axis = extr_data['at_x_axis']

        ax.plot(at_x_axis, mae_progress, label=extr_name + f' (MAE of fit = {round(mae_progress[-1], 2)})')
    
    ax.set_xlabel('Time spent fitting the parametric model [s]')
    ax.set_ylabel('MAE')
    
    ax.legend()
    ax.set_title('MAE progress on partial learning curve')

    plt.savefig(os.path.join('plot_thesis_figures', f'MAE_progress_{save_name_suffix}.png'), dpi=300)
    plt.show()




def get_plot_with_all_extrapolators_fitting_on_averaged_out_curve(extrapolators: list[LearningCurveExtrapolator], save_name_suffix: str, EPOCHS_PARTIAL_LC: int = 20):
    """
    Parameters
    ----------
    extrapolators : list[LearningCurveExtrapolator]
        Extrapolators used in experiment
    
    save_name_suffix : str
        suffix to append to the save name (three/all)
    """
    extrapolators = get_top_three_extrapolators(lr=0.1)#get_fresh_extrapolators(lr=0.1)
    lcs_data, avg_lc_data = get_learning_curves_data(15625)

    FIT_FOR_TIME = (EPOCHS_PARTIAL_LC * 20) / len(extrapolators) # Time in seconds for which if we train every extrapolator we match time required for EPOCHS_PARTIAL_LC
    
    extrapolators_results = {}

    for extrapolator in extrapolators:
        print(f'Started {extrapolator.get_name()}.', end=' ')
        start = time()

        at_x_axis, progress_dict, mae_progress = extrapolator.fit_data(
            avg_lc_data[:EPOCHS_PARTIAL_LC], 
            return_variables_progress=True, 
            time_seconds=FIT_FOR_TIME
        )
        
        end = time()
        print(f'Took {end - start} seconds')

        # Save the progress of extrapolator
        extrapolators_results[extrapolator.get_name()] = {'at_x_axis': at_x_axis, 'progress_dict': progress_dict, 'mae_progress': mae_progress, 'time': end - start, 'extrapolator': extrapolator}
        

    
    plot_progress_of_extrapolators(extrapolators_results, avg_lc_data, save_name_suffix=save_name_suffix, EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC)

########################################
############# Finetuning test ##########
########################################

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


def finetune_standard_approach(path_to_fig_dir: str, EPOCHS_PARTIAL_LC: int = 20):

    lcs_data, avg_lc_data = get_learning_curves_data(50)
    tested_lrs = [0.5, 0.2, 0.1, 0.05, 0.01]
    TIME_PER_EXTRAPOLATOR = 20 # 2 epochs worth of time [seconds]
    
    fig, axes = plt.subplots(1, len(tested_lrs), layout='constrained', figsize=(len(tested_lrs) * 10, 8))

    # For each tested lr
    for i, lr in enumerate(tested_lrs):
        MAE_list = []

        # For each architecture
        for full_lc in lcs_data:
            train_data = full_lc[:EPOCHS_PARTIAL_LC]

            extr_ensemble = LearningCurveExtrapolatorsEnsembler(get_top_three_extrapolators(lr=lr))
            extr_ensemble.fit_extrapolators(train_data, time_seconds=TIME_PER_EXTRAPOLATOR) # 2 epochs worth of time [seconds]

            mae_of_fit = get_MAE_of_fit(train_data, [extr_ensemble.predict_avg(i + 1) for i in range(EPOCHS_PARTIAL_LC)])
            MAE_list.append(mae_of_fit)

        final_avg_MAE = np.average(MAE_list)
        
        # Now plot the graph having the final MAE and also plot fit of the avg curve
        extr_ensemble = LearningCurveExtrapolatorsEnsembler(get_top_three_extrapolators(lr=lr))
        extr_ensemble.fit_extrapolators(avg_lc_data[:EPOCHS_PARTIAL_LC], time_seconds=TIME_PER_EXTRAPOLATOR) 
        
        plot_ensembler_prediction_on_data(axes[i], extr_ensemble, avg_lc_data, EPOCHS_PARTIAL_LC, f'For lr = {lr} we have average MAE of fit = {round(final_avg_MAE, 5)}')
    
        # And also save as singular file
        fig_singular, ax_singular = plt.subplots(layout='constrained', figsize=(10, 8))
        plot_ensembler_prediction_on_data(ax_singular, extr_ensemble, avg_lc_data, EPOCHS_PARTIAL_LC, f'For lr = {lr} we have average MAE of fit = {round(final_avg_MAE, 5)}')
        fig_singular.savefig(os.path.join(path_to_fig_dir, f'Standart_init_approach_lr_{lr}.png'))


    # Save the resulting plot
    if not os.path.exists(path_to_fig_dir):
        os.mkdir(path_to_fig_dir)

    fig.suptitle('Standard initialization approach', fontsize=24)
    fig.savefig(os.path.join(path_to_fig_dir, 'Standart_init_approach.png'))

def finetune_average_fit_approach(path_to_fig_dir: str, EPOCHS_PARTIAL_LC: int = 20):

    lcs_data, avg_lc_data = get_learning_curves_data(50)
    tested_lrs_tuples = [(0.1, 0.01), (0.1, 0.001)]
    tested_time_splits = [0.05, 0.15]
    TIME_PER_EXTRAPOLATOR = 20 * 2 # 2 epochs worth of time [seconds]

    fig, axes = plt.subplots(len(tested_time_splits), len(tested_lrs_tuples),  layout='constrained', figsize=(len(tested_time_splits) * 10, len(tested_lrs_tuples) * 8))

    OVERALL_TIME_PER_EXTRAPOLATOR = TIME_PER_EXTRAPOLATOR * len(lcs_data)

    for i, (lr_average_lc, lr_concrete_lc) in enumerate(tested_lrs_tuples):
        for j, (frac_time_spent_on_average_lc) in enumerate(tested_time_splits):
            
            # Do the training on averaged curve first
            pass # Add copy to LCEEnsembler and LCE...

            # Then go by each concrete learning curve
            # Here simply copy Ensembler and fit it additionaly

        

    



def main(path_to_fig_dir: str):
    # For MAE_progress_all.png and predictions_x_ground_truth_all.png
    #get_plot_with_all_extrapolators_fitting_on_averaged_out_curve(get_fresh_extrapolators(0.1), save_name_suffix='all')

    # For MAE_progress_three.png and predictions_x_ground_truth_three.png
    #get_plot_with_all_extrapolators_fitting_on_averaged_out_curve(get_top_three_extrapolators(0.1), save_name_suffix='three')
     
    # Find out best hyper parameters for naive initialization approach
    finetune_standard_approach(os.path.join(path_to_fig_dir, 'Init_approaches_finetune'))


    
    


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)  
    main('plot_thesis_figures')