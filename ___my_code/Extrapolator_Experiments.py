import os
import random
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LearningCurveExtrapolator import *
from utils import ArchiSampler, load_data, calc_spearman_rank_correlation_coef
from target_extrapolations import standart_initialization_extrapolation, prefit_avg_initialization_extrapolation


def get_learning_curves_data(amount_of_archs: int, also_return_arch_i: bool = False) -> tuple[list[list[float]], list[float]]:
    """
    Uses np.seed to base its selection.

    Returns whole learning curves of `amount_of_archs` random architectures and averaged out learning curve
    """
    data = load_data()
    assert len(data) == 15625, 'Missing architectures in _clean_all_merged.csv'
    
    sampled_archs_i = []
    arch_i_sampler = ArchiSampler(len(data))    

    whole_lcs = []
    avg_whole_lc = np.zeros(200)
    for _ in range(amount_of_archs):
        arch_i = arch_i_sampler.sample()
        val_accs = np.array(data[arch_i]['val_accs'])

        avg_whole_lc += val_accs
        whole_lcs.append(val_accs)

        sampled_archs_i.append(arch_i)

    avg_whole_lc = avg_whole_lc / amount_of_archs
        
    if also_return_arch_i:
        return whole_lcs, avg_whole_lc, sampled_archs_i
    else:
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


def finetune_standard_approach(lcs_data: list[list[float]], avg_lc_data: list[float], plots_path: str, TIME_PER_EXTRAPOLATOR_PER_LC: float, EPOCHS_PARTIAL_LC: int = 20):

    tested_lrs = [1, 0.5, 0.2, 0.1, 0.05, 0.01]

    for i, lr in enumerate(tested_lrs):
        print('-----------------------------------')
        print(f'Testing setting {i+1}/{len(tested_lrs)}')
        print('-----------------------------------')

        val_accs_predictions = standart_initialization_extrapolation(
            lcs_data,
            avg_lc_data,
            TIME_PER_EXTRAPOLATOR_PER_LC,
            get_top_three_extrapolators,
            lr,
            EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC,
            plots_path=plots_path,
            verbose=True
        )

        #print(val_accs_predictions)

def prefit_average_fit_approach(lcs_data: list[list[float]], avg_lc_data: list[float], plots_path: str, TIME_PER_EXTRAPOLATOR_PER_LC: float, EPOCHS_PARTIAL_LC: int = 20):

    tested_lrs_tuples = [(0.2, 0.05), (0.1, 0.01), (0.1, 0.001)]
    tested_time_splits = [0.05, 0.15, 0.3]

    i, total_i = 0, len(tested_lrs_tuples) * len(tested_time_splits)
    for lrs_tuple in tested_lrs_tuples:
        for frac_time_spent_on_average_lc in tested_time_splits:
            i += 1
            print('-----------------------------------')
            print(f'Testing setting {i}/{total_i}')
            print('-----------------------------------')

            val_accs_predictions = prefit_avg_initialization_extrapolation(
                lcs_data,
                avg_lc_data,
                TIME_PER_EXTRAPOLATOR_PER_LC,
                get_top_three_extrapolators,
                lrs_tuple,
                frac_time_spent_on_average_lc,
                EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC,
                plots_path=plots_path,
                verbose=True
            )

            #print(val_accs_predictions)
            
def plot_Spearman_rank_corr_coef_for_initialization_approaches(lcs_data: list[list[float]], avg_lc_data: list[float], sampled_archs_i: list[int], plots_path: str, TIMES_PER_EXTRAPOLATOR_PER_LC: list[float], EPOCHS_PARTIAL_LC: int = 20):
    """
    Plots Spearman-rank-correlation-coeficient against time-per-extrapoaltion-per-lc for 
    Standart and prefit average initialization approaches with their best hyper parameters as in thesis
    """
    LCS_AMOUNT = len(lcs_data)

    # Hyper parameters for Standart approach
    standart_aproach___lr = 0.2

    # Hyper parameters for Prefit average approach
    prefit_average___lrs = (0.2, 0.05)
    prefit_average___frac_time_spent_prefiting = 0.05

    # And also create ground truth dict used for calculating Spearman rank corr coef
    ground_truth_arch_dict = {
        arch_i: val_accs[-1] for (arch_i, val_accs) in zip(sampled_archs_i, lcs_data)
    }

    # Spearmans accumulations
    standart_aproach___spearmans = []
    prefit_average___spearmans = []

    # For every time budget
    for TIME_PER_EXTRAPOLATOR_PER_LC in TIMES_PER_EXTRAPOLATOR_PER_LC:
        print()
        print(f'Starting Standart initialization approach for time budget: {TIME_PER_EXTRAPOLATOR_PER_LC}')
        print()

        # Run standart approach for predicitons
        standart_aproach___val_accs_predictions = standart_initialization_extrapolation(
            lcs_data,
            avg_lc_data,
            TIME_PER_EXTRAPOLATOR_PER_LC,
            get_top_three_extrapolators,
            standart_aproach___lr,
            EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC,
            plots_path=None,
            verbose=True
        )
        # Create dict of val accs predictions for Spearman rank corr coef calculation
        standart_aproach___val_accs_predictions = {
            arch_i: predicted_val_acc for (arch_i, predicted_val_acc) in zip(sampled_archs_i, standart_aproach___val_accs_predictions)
        }

        # Get its resulting Spearman
        standart_aproach___spearmans.append(
            calc_spearman_rank_correlation_coef(ground_truth_arch_dict, standart_aproach___val_accs_predictions, also_plot=False)
        )
        
        # Now do the same for prefit average approach
        print()
        print(f'Starting Prefit average initialization approach for time budget: {TIME_PER_EXTRAPOLATOR_PER_LC}')
        print()

        prefit_average___val_accs_predictions = prefit_avg_initialization_extrapolation(
            lcs_data,
            avg_lc_data,
            TIME_PER_EXTRAPOLATOR_PER_LC,
            get_top_three_extrapolators,
            prefit_average___lrs,
            prefit_average___frac_time_spent_prefiting,
            EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC,
            plots_path=None,
            verbose=True
        )
        # Create dict of val accs predictions for Spearman rank corr coef calculation
        prefit_average___val_accs_predictions = {
            arch_i: predicted_val_acc for (arch_i, predicted_val_acc) in zip(sampled_archs_i, prefit_average___val_accs_predictions)
        }
        # Get its resulting Spearman
        prefit_average___spearmans.append(
            calc_spearman_rank_correlation_coef(ground_truth_arch_dict, prefit_average___val_accs_predictions, also_plot=False)
        )
        
    
    # After we have collected all the data lets plot them
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))
    print(TIMES_PER_EXTRAPOLATOR_PER_LC)

    # Plot Standart's approach Spearmans
    ax.plot(TIMES_PER_EXTRAPOLATOR_PER_LC, standart_aproach___spearmans, 'o--', label='Standart approach')
    print(standart_aproach___spearmans)
    # And also the prefit average's approach Spearmans
    ax.plot(TIMES_PER_EXTRAPOLATOR_PER_LC, prefit_average___spearmans, 'o--', label='Prefit average approach')
    print(prefit_average___spearmans)

    ax.set_ylabel('Spearman rank correlation coefficient')
    ax.set_xlabel('Time budget per extrapolator per partial learning curve [s]')
    ax.legend()

    ax.set_title(f'Rankings of {LCS_AMOUNT} architectures depending on the time budget')

    os.makedirs(plots_path, exist_ok=True)
    plt.savefig(os.path.join(plots_path, f'Spearman_rank_corr_coef_comparison.png'), dpi=300)
            



def check_statistics_of_prefit_average(lcs_data: list[list[float]], avg_lc_data: list[float], sampled_archs_i: list[int], EPOCHS_PARTIAL_LC: int = 20):
        from scipy import stats

        ground_truth = [val_accs[-1] for val_accs in lcs_data]

        custom_ground_truth = {
            arch_i: val_accs[-1] for (arch_i, val_accs) in zip(sampled_archs_i, lcs_data)
        }

        
        print('prefit avg')
        prefit_average___val_accs_predictions = prefit_avg_initialization_extrapolation(
            lcs_data,
            avg_lc_data,
            5,
            get_top_three_extrapolators,
            (0.2, 0.05),
            0.05,
            EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC,
            plots_path=os.path.join('plot_thesis_figures', 'one_k_check'),
            verbose=True
        )

        

        ken_tau = stats.kendalltau(prefit_average___val_accs_predictions, ground_truth)
        print(f'kendalltau: {ken_tau.statistic}')

        spearman_coef = stats.spearmanr(prefit_average___val_accs_predictions, ground_truth)
        print(f'spearmanr: {spearman_coef.statistic}')

        custom_predictions = {
            arch_i: val_acc for (arch_i, val_acc) in zip(sampled_archs_i, prefit_average___val_accs_predictions)
        }
        plt.show()
        custom_spearman = calc_spearman_rank_correlation_coef(custom_ground_truth, custom_predictions, also_plot=False)
        print(f'custom_spearman: {custom_spearman}')

        print('standart')
        standart_aproach___val_accs_predictions = standart_initialization_extrapolation(
            lcs_data,
            avg_lc_data,
            20,
            get_top_three_extrapolators,
            0.2,
            EPOCHS_PARTIAL_LC=EPOCHS_PARTIAL_LC,
            plots_path=os.path.join('plot_thesis_figures', 'one_k_check'),
            verbose=True
        )

        ken_tau = stats.kendalltau(standart_aproach___val_accs_predictions, ground_truth)
        print(f'kendalltau: {ken_tau.statistic}')

        spearman_coef = stats.spearmanr(standart_aproach___val_accs_predictions, ground_truth)
        print(f'spearmanr: {spearman_coef.statistic}')

        custom_predictions = {
            arch_i: val_acc for (arch_i, val_acc) in zip(sampled_archs_i, standart_aproach___val_accs_predictions)
        }
        plt.show()
        custom_spearman = calc_spearman_rank_correlation_coef(custom_ground_truth, custom_predictions, also_plot=False)
        print(f'custom_spearman: {custom_spearman}')











def main(path_to_fig_dir: str):
    """
    # For MAE_progress_all.png and predictions_x_ground_truth_all.png
    get_plot_with_all_extrapolators_fitting_on_averaged_out_curve(get_fresh_extrapolators(0.1), save_name_suffix='all')
    #"""

    """
    # For MAE_progress_three.png and predictions_x_ground_truth_three.png
    get_plot_with_all_extrapolators_fitting_on_averaged_out_curve(get_top_three_extrapolators(0.1), save_name_suffix='three')
    #"""
    LCS_AMOUNT = 50
    lcs_data, avg_lc_data = get_learning_curves_data(LCS_AMOUNT)

    """
    # Find out best hyper parameters for standart initialization approach
    finetune_standard_approach(
        lcs_data, 
        avg_lc_data,
        os.path.join(path_to_fig_dir, 'Init_approaches_finetune', 'Standart_approach'),
        TIME_PER_EXTRAPOLATOR_PER_LC=20,
    )
    #"""

    """
    # Fing out the best hyper parameters for prefitting average lc initialization approach
    prefit_average_fit_approach(
        lcs_data, 
        avg_lc_data,
        os.path.join(path_to_fig_dir, 'Init_approaches_finetune', 'Prefit_avg_approach'),
        TIME_PER_EXTRAPOLATOR_PER_LC=20
    )    
    #"""
    

    LCS_AMOUNT = 1000
    lcs_data, avg_lc_data, sampled_archs_i = get_learning_curves_data(LCS_AMOUNT, also_return_arch_i=True)
    """
    plot_Spearman_rank_corr_coef_for_initialization_approaches(
        lcs_data, 
        avg_lc_data, 
        sampled_archs_i,
        os.path.join(path_to_fig_dir, 'Init_approaches_Spearman_comparison'),
        TIMES_PER_EXTRAPOLATOR_PER_LC=[1, 5, 10, 20, 40] 
    )
    #"""

    #check_statistics_of_prefit_average(lcs_data, avg_lc_data, sampled_archs_i)




if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)  
    main('plot_thesis_figures')