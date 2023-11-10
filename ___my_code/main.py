import os
from time import time

import pandas as pd
import numpy as np
import xgboost

from utils import *
from LearningCurveExtrapolator import *
from ArchitectureEncoder import *
from PredictorDataGetter import *

class RunConfig:
    def __init__(
            self, 
            epochs_trained_per_arch_for_extrapolatos: int, 
            secs_per_extrapolator_fitting: int, 
            num_wanted_architectures: int, 
            get_lc_extrapolators_ensembler, 
            data_getter: DataGetter,
            POP_SIZE: int,
            SAMPLE_SIZE: int,
            EVOLVE_CYCLES: int
        ) -> None:
        """
        Parameters
        ----------
        epochs_trained_per_arch_for_extrapolatos : int
            The number of training epochs of architectures on which validation accuracies the extrapolators will be fitted
            Note: Must be in range [1,200]
        
        secs_per_extrapolator_fitting : int
            The amount of seconds each extrapolator will have for fitting the partial learning curve of the architecture

        num_wanted_architectures : int
            The amount of architectures for which we will extrapolate theyr final validation accuracy. In other words,
            this is the number of architectures on which the performance predictor will be trained

        get_lc_extrapolators_ensembler: func: None -> LearningCurveExtrapolatorsEnsembler 
            Function which returns the list of Extrapoltors which will be used in LearningCurveExtrapolatorsEnsembler

        data_getter : DataGetter
            Data getter which also defines architecture encoding used

        POP_SIZE : int
            Defines population size for the AE.

        SAMPLE_SIZE : int
            Sample size from which parent for the child in the AE will be chosen.

        EVOLVE_CYCLES : int
            Cycles for which the AE will run
        """
        self.epochs_trained_per_arch_for_extrapolatos = epochs_trained_per_arch_for_extrapolatos
        self.secs_per_extrapolator_fitting = secs_per_extrapolator_fitting
        self.num_wanted_architectures = num_wanted_architectures
        self.get_lc_extrapolators_ensembler = get_lc_extrapolators_ensembler
        self.data_getter = data_getter
        self.POP_SIZE = POP_SIZE
        self.SAMPLE_SIZE = SAMPLE_SIZE
        self.EVOLVE_CYCLES = EVOLVE_CYCLES

    def get_expected_time_for_perf_perdictor_data_gathering(self) -> float:
        """
        Returns the expected amount of secods representing how long the data gathering
        process for performance predictor training will take
        """
        num_of_extrapolators_in_ensembler = len(self.get_lc_extrapolators_ensembler().show_extrapolators_list())
        return (self.epochs_trained_per_arch_for_extrapolatos * Constants.AVG_TIME_PER_ARCH_TRAINING_EPOCH + self.secs_per_extrapolator_fitting * num_of_extrapolators_in_ensembler) * self.num_wanted_architectures
    
    def get_new_lc_extrapolators_ensembler(self) -> LearningCurveExtrapolatorsEnsembler:
        """
        Returns new LearningCurveExtrapolatorsEnsembler defined by 
        `get_lc_extrapolators_ensembler` parameter supplied to constructor 
        """
        return self.get_lc_extrapolators_ensembler()



def get_training_data_for_predictor(config: RunConfig, verbose: bool = True, add_arch_i_and_real_val_acc: bool = False) -> tuple[list[list[float]], list[float]]:
    """
    Creates and returns training data for performance predictor
    """
    # Create ArchiSampler with array of arch_i for all architectures shuffled randomly
    arch_i_sampler = ArchiSampler(config.data_getter.get_amount_of_architectures())

    # Here we will collect data for predictor's training
    data_xs, data_ys = [], []
    if add_arch_i_and_real_val_acc:
        data_arch_i, data_real_valaccs = [], []

    beg_training_time = time()
    while time() - beg_training_time < config.get_expected_time_for_perf_perdictor_data_gathering():

        arch_i = arch_i_sampler.sample()
        if verbose:
            print(f'Getting training data for arch_i {arch_i}')

        data_x, data_y = config.data_getter.get_training_data_by_arch_index(
            arch_i,
            (config.get_new_lc_extrapolators_ensembler(), config.epochs_trained_per_arch_for_extrapolatos, config.secs_per_extrapolator_fitting)
        )

        data_xs.append(data_x)
        data_ys.append(data_y)

        if add_arch_i_and_real_val_acc:
            data_arch_i.append(arch_i)
            data_real_valaccs.append(config.data_getter.get_real_val_acc_of_arch(arch_i))

    if verbose:    
        print(f'We expected to get {config.num_wanted_architectures} datapoints and got {len(data_xs)}.')

    if add_arch_i_and_real_val_acc:
        return data_xs, data_ys, data_arch_i, data_real_valaccs
    else:
        return data_xs, data_ys



def main(config: RunConfig):
    np.random.seed(420)
    data_xs, data_ys = get_training_data_for_predictor(config)
    
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(data_xs, data_ys)


    ###
    ### Now we run the AE algo
    ###

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


    POP_SIZE = config.POP_SIZE
    SAMPLE_SIZE = config.SAMPLE_SIZE
    EVOLVE_CYCLES = config.EVOLVE_CYCLES

    # Architecture is represented by PopulationElement
    population: list[PopulationElement] = []
    # Every architecture we encountered is in hisotry
    history:    list[PopulationElement] = []

    # Fill the initial population
    for _ in range(POP_SIZE):
        # Get random's architecture features
        arch_i = np.random.randint(config.data_getter.get_amount_of_architectures())        
        arch_i_features = config.data_getter.get_prediction_features_by_arch_index(arch_i)

        # Predict it's final validation accuracy
        predicted_val_acc = regr.predict([arch_i_features])
        # And also get it's real validation accuracy from the NasBench201 data
        real_val_acc = config.data_getter.get_real_val_acc_of_arch(arch_i)

        # Add such element to both population and history
        pop_elem = PopulationElement(arch_i, predicted_val_acc, real_val_acc)
        population.append(pop_elem)
        history.append(pop_elem)

    # Now that we have filled the population with POP_SIZE elements, let's start the AE loop
    for evolve_cycle in range(EVOLVE_CYCLES):

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
        parent_std_encoding = config.data_getter.get_std_encoding_of_arch_i(parent.arch_i)

        # Now mutate parent in order to get it's child
        child_std_encoding = mutate(parent_std_encoding)
        # TODO: Kedze este nemam vsetky data, tak sa moze stat ze dieta neni v datach...
        # Zatial teda zoberies nahodnu architekturu s tych 200 generovanych
        try:
            child_arch_i = config.data_getter.get_arch_i_from_standart_encoding(child_std_encoding)
        except:
            child_arch_i = np.random.randint(config.data_getter.get_amount_of_architectures())

        # Predict child's final validation accuracy
        child_predict_feats = config.data_getter.get_prediction_features_by_arch_index(child_arch_i)
        child_predicted_val_acc = regr.predict([child_predict_feats])
        # And also get child's real validation accuracy from the NasBench201 data
        child_real_val_acc = config.data_getter.get_real_val_acc_of_arch(child_arch_i)

        # Add child to population and history
        child_pop_elem = PopulationElement(child_arch_i, child_predicted_val_acc, child_real_val_acc)
        population.append(child_pop_elem)
        history.append(child_pop_elem)

        # Remove oldest (index 0) PopulationElement from population
        population.pop(0)

    plot_history(history, POP_SIZE)

    


def test_training_data(config: RunConfig):
    np.random.seed(420)
    data_xs, data_ys, data_arch_i, data_real_valaccs = get_training_data_for_predictor(config, add_arch_i_and_real_val_acc=True)
    
    #from sklearn.ensemble import RandomForestRegressor
    #regr = RandomForestRegressor(max_depth=2, random_state=0)
    #regr.fit(data_xs, data_ys)

    #predicted_val_accs = regr.predict(data_xs)

    data_real = {arch_i: real_valacc for (arch_i, real_valacc) in zip(data_arch_i, data_real_valaccs)}
    data_predicted = {arch_i: pred_valacc for (arch_i, pred_valacc) in zip(data_arch_i, data_ys)}
    
    # TODO check if it really works
    # https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    srcc = calc_spearman_rank_correlation_coef(data_real, data_predicted, also_plot=True)
    



if __name__ == '__main__':

    def get_lc_extrapolators_ensembler():
        return LearningCurveExtrapolatorsEnsembler([LogShiftScale_Extrapolator(), VaporPressure_Extrapolator()])
    
    data_getter = DataGetter(
        OneHotOperation_Encoder(),
        ['jacob_cov', 'grad_norm']
    )

    config = RunConfig(
        epochs_trained_per_arch_for_extrapolatos=20,
        secs_per_extrapolator_fitting=40,
        num_wanted_architectures=10,
        get_lc_extrapolators_ensembler=get_lc_extrapolators_ensembler,
        data_getter=data_getter,
        POP_SIZE=10,
        SAMPLE_SIZE=3,
        EVOLVE_CYCLES=500
    )

    test_training_data(config)
    #main(config)