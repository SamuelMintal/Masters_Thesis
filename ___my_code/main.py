import os
from time import time
from functools import partial
import random
from collections import defaultdict

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy import stats


from utils import *
from LearningCurveExtrapolator import *
from ArchitectureEncoder import *
from PredictorDataGetter import *
from target_extrapolations import prefit_avg_initialization_extrapolation


def get_optimal_prefit_avg_partialed_func() -> Callable:
    """
    returns prefit_avg_initialization_extrapolation function with hyperparameters
    being fixed to the optimal values as found by Extrapolations Experiments only
    requiring 2 inputs:
        1st being the list of lcs (uses only first 20 epochs for fitting as done in thesis)
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



def run_evolution(perf_predictor: XGBRegressor, data_getter: DataGetter, POP_SIZE: int, SAMPLE_SIZE: int, EVOLVE_CYCLES: int, verbose_each_evol_cycles: int = 500) -> list[float]:
    """
    Runs Aeging evolution algorithm returning history

    Parameters
    ----------
    perf_predictor: XGBRegressor
        Trained XGBoost predictor used for guiding the evolution
     
    data_getter: DataGetter
        DataGetter which is used for acquiring following data:
            1. get_amount_of_architectures
            2. get_prediction_features_by_arch_index
            3. get_real_val_acc_of_arch
            4. get_std_encoding_of_arch_i
            5. get_arch_i_from_standart_encoding

    verbose_each_evol_cycles : int, default=500

    """

    def mutate(parent_std_encoding: list[int]) -> list[int]:
        """
        TODO: ability to add following thing:
        1. Chance that the arch will not be mutated at all
            (Then make sure that when you mutate you do not get the same arch)
        """
        # Copy parents encoding so we do not change it
        child_std_encoding = parent_std_encoding[:]

        what_op_mutate = np.random.randint(len(child_std_encoding))
        to_what_mutate = np.random.randint(5) # There are 5 operations possible in std encoding

        child_std_encoding[what_op_mutate] = to_what_mutate

        return child_std_encoding


    # Architecture is represented by PopulationElement
    population: list[PopulationElement] = []
    # Every architecture we encountered is in hisotry
    history:    list[PopulationElement] = []

    # Fill the initial population
    for _ in range(POP_SIZE):
        # Get random's architecture features
        arch_i = np.random.randint(data_getter.get_amount_of_architectures())        
        arch_i_features = data_getter.get_prediction_features_by_arch_index(arch_i)

        # Predict it's final validation accuracy
        predicted_val_acc = perf_predictor.predict([arch_i_features])[0]
        # And also get it's real validation accuracy from the NasBench201 data
        real_val_acc = data_getter.get_real_val_acc_of_arch(arch_i)

        # Add such element to both population and history
        pop_elem = PopulationElement(arch_i, predicted_val_acc, real_val_acc)
        population.append(pop_elem)
        history.append(pop_elem)

    # Now that we have filled the population with POP_SIZE elements, let's start the AE loop
    for evolve_cycle in range(EVOLVE_CYCLES):
        if (evolve_cycle + 1) % verbose_each_evol_cycles == 0:
            print(f'Starting evolution cycle {evolve_cycle + 1}/{EVOLVE_CYCLES}')

        # First we need to fill sample set
        sample: list[PopulationElement] = []
        for _ in range(SAMPLE_SIZE):
            # Add random PopulationElement from the popultaion to sample
            # Note: the PopulationElement stays in the population
            sample.append(population[np.random.randint(POP_SIZE)])

        # Select parent as the PopulationElement from the sample with
        # the biggest predicted validation accuracy
        parent_i = np.argmax([candidate.predicted_val_acc for candidate in sample])
        parent = sample[parent_i]
        parent_std_encoding = data_getter.get_std_encoding_of_arch_i(parent.arch_i)

        # Now mutate parent in order to get it's child
        child_std_encoding = mutate(parent_std_encoding)
        child_arch_i = data_getter.get_arch_i_from_standart_encoding(child_std_encoding)

        # Predict child's final validation accuracy
        child_predict_feats = data_getter.get_prediction_features_by_arch_index(child_arch_i)
        child_predicted_val_acc = perf_predictor.predict([child_predict_feats])[0]
        # And also get child's real validation accuracy from the NasBench201 data
        child_real_val_acc = data_getter.get_real_val_acc_of_arch(child_arch_i)

        # Add child to population and history
        child_pop_elem = PopulationElement(child_arch_i, child_predicted_val_acc, child_real_val_acc)
        population.append(child_pop_elem)
        history.append(child_pop_elem)

        # Remove oldest (index 0) PopulationElement from population
        population.pop(0)

    return history
    


    

def run_evolution_experiments(TRAIN_SET_SIZE: int = 300, RUNS_PER_SETTING: int = 5):
    """
    Runs AE algorithm for various settings

    Parameters
    ----------
    TRAIN_SET_SIZE : int, default=300
        What is the train set size for the predictor to learn on
    
    RUNS_PER_SETTING : int, default=5
        How many times is singular setting ran
    """
    evolution_runs_times = []
    runs_histories = defaultdict(list)

    data_getter_ZC = get_optimal_data_getter(OneHotOperation_Encoder())

    data_getter_NOT_ZC = get_optimal_data_getter(OneHotOperation_Encoder())
    data_getter_NOT_ZC.features_to_get = []

    # AE Hyperparameters
    POP_SIZE: int = 100
    SAMPLE_SIZE: int = 25
    EVOLVE_CYCLES: int = 1000

    use_ZC = [True, False]
    use_Extrapolated_targets = [True, False]

    for cur_run in range(RUNS_PER_SETTING):
        print(f'Started run {cur_run + 1}/{RUNS_PER_SETTING}')
        # Get architectures which will be used for traning this run
        arch_i_sampler = ArchiSampler(data_getter_ZC.get_amount_of_architectures())
        train_archs_i = [arch_i_sampler.sample() for _ in range(TRAIN_SET_SIZE)]

        ###
        ### And now get all possible training data variations
        ###
        # Start with inclusion of ZC proxies and extrapolated targets
        train_inputs_WITH_ZC, train_targets_EXTRAPOLATED = data_getter_ZC.get_training_data_by_arch_indices(train_archs_i)

        # Get Features WITHOUT ZC proxeis and REAL targets
        train_inputs_NOT_ZC = np.array([data_getter_NOT_ZC.get_prediction_features_by_arch_index(arch_i) for arch_i in train_archs_i])
        train_targets_REAL = np.array([data_getter_NOT_ZC.get_real_val_acc_of_arch(arch_i) for arch_i in train_archs_i])
        
        # Now for all 4 combinations of features
        for zc_usage in use_ZC:
            for extrap_targets_usage in use_Extrapolated_targets:

                setting_str = f"XGBoost (using ZC proxies: {'Yes' if zc_usage else 'No'} | Extrapolated train targets: {'Yes' if extrap_targets_usage else 'No'})"

                # Select current inputs/outputs for the model to be learned
                cur_train_inputs = train_inputs_WITH_ZC if zc_usage else train_inputs_NOT_ZC
                cur_train_outputs = train_targets_EXTRAPOLATED if extrap_targets_usage else train_targets_REAL

                # Train the predictor
                XGBoost_predictor = XGBRegressor()
                XGBoost_predictor.fit(cur_train_inputs, cur_train_outputs)

                cur_data_getter = data_getter_ZC if zc_usage else data_getter_NOT_ZC

                print(f'Starting evolution: {setting_str}')
                t_start = time()
                history = run_evolution(XGBoost_predictor, cur_data_getter, POP_SIZE, SAMPLE_SIZE, EVOLVE_CYCLES)
                t_took = round(time() - t_start, 1)
                print(f'Took {t_took} seconds')
                evolution_runs_times.append(t_took)

                runs_histories[setting_str].append(history)

    plot_histories(runs_histories, POP_SIZE)
    print(evolution_runs_times)



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    run_evolution_experiments(TRAIN_SET_SIZE=300, RUNS_PER_SETTING=30)