import os
from time import time

import pandas as pd
import numpy as np
import xgboost

from utils import *
from LearningCurveExtrapolator import *
from ArchitectureEncoder import *
from PredictorDataGetter import *
        

def main(arch_encoder: ArchitectureEncoder, features_to_get: list[str], get_chosen_extrapolators):
    np.random.seed(42)

    TRAIN_FOR_SECONDS = 100
    NUM_ARCH_TRAIN_EPOCHS_TO_FIT_EXTRAPOLATOR = 20
    SECS_PER_EXTRAPOLATOR_FITTING = 10

    ###
    ### 1st we do the training of the performance predictor
    ###

    # Create PredictorDataGetter which will be used
    # for getting data for predictor
    predictor_data_getter = PredictorDataGetter()

    # Create ArchiSampler with array of arch_i for all architectures shuffled randomly
    arch_i_sampler = ArchiSampler(predictor_data_getter.get_amount_of_architectures())

    # Here we will collect data for predictor's training
    data_xs, data_ys = [], []

    beg_training_time = time()
    while time() - beg_training_time < TRAIN_FOR_SECONDS:

        arch_i = arch_i_sampler.sample()

        data_x, data_y = predictor_data_getter.get_data_by_arch_index(
            arch_i,
            arch_encoder,
            (LearningCurveExtrapolatorsEnsembler(get_chosen_extrapolators(), verbose=False), NUM_ARCH_TRAIN_EPOCHS_TO_FIT_EXTRAPOLATOR, SECS_PER_EXTRAPOLATOR_FITTING),
            features_to_get
        )

        data_xs.append(data_x)
        data_ys.append(data_y)
        
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(data_xs, data_ys)
        

    


def test():
    pred_data_getter = PredictorDataGetter()

    data_xs, data_ys = [], []
    for arch_i in range(10):
        data_x, data_y = pred_data_getter.get_data_by_arch_index(
            arch_i,
            OneHotOperation_Encoder(),
            (LearningCurveExtrapolatorsEnsembler([LogShiftScale_Extrapolator(), VaporPressure_Extrapolator()], verbose=False), 20, 10),
            ['jacob_cov', 'grad_norm']
        )

        data_xs.append(data_x)
        data_ys.append(data_y)

    print(data_xs)
    print(data_ys)

    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(data_xs[:-1], data_ys[:-1])

    predicted, real = regr.predict([data_xs[-1]]), data_ys[-1]
    print(f'predicted: {predicted}, real: {real}')



if __name__ == '__main__':
    test()
    #main()