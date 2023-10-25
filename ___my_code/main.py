import os

import pandas as pd
import numpy as np
import xgboost

from utils import load_data
from LearningCurveExtrapolator import *
from ArchitectureEncoder import  *

class PredictorDataGetter:
    """
    Class for constructing training data for the Performance predictor.
    """
    def __init__(self, data_path: str = os.path.join('___my_code', '_clean_all_merged.csv')) -> None:
        self.data = load_data(data_path)

    def get_arch_indices(self) -> list[int]:
        """
        Returns all indices of available architectures
        """
        return list(self.data.keys())

    def get_data_by_arch_index(
            self,
            arch_index: int, 
            arch_encoder: ArchitectureEncoder, 
            lc_extrapolaion_info: tuple[LearningCurveExtrapolatorsEnsembler, int, int],
            features_to_get: list[str]
        ) -> tuple[list[int], int]:
        """
        Returns input features as well as extrapolated target for specified architecture.

        Parameters
        ----------
        arch_index: int 
            Index of an architecture for which to retrieve the data (arch_i).

        arch_encoder : ArchitectureEncoder
            Architecture encoder which to use for encoding the architecture

        lc_extrapolaion_info : tuple[LearningCurveExtrapolatorsEnsembler, int, int]
            Tuple of 3 elements containing all of the information about target generation:
            - LearningCurveExtrapolatorsEnsembler used for extrapolating target.            
            - Amount of architecture's training epochs on which the extrapolator will be fitted.
            - The amount of seconds for which each extrapolator in ensembler will be able to do it's fitting.

        features_to_get : list[str]
            list of features which to retrieve.

        Returns
        -------
        tuple[list[int], int]
            tuple containing training-data and it's target
        """
        # Firstly get architecture's data
        arch_data = self.data[arch_index]
        
        # Secondly get architecture's encoding
        arch_encoding = arch_encoder.convert_from_arch_string(arch_data['arch_str'])

        # Thirdly collect the required features
        arch_features = [arch_data[feat] for feat in features_to_get]
        
        # And lastly before we return results, get the target
        # validation accuracy of the architecture
        lc_extrapolator, n_training_data, n_training_secs = lc_extrapolaion_info
        trainning_data = arch_data['val_accs'][:n_training_data]
        lc_extrapolator.fit_extrapolators(trainning_data, time_seconds=n_training_secs)
        target = lc_extrapolator.predict_avg(200).numpy()

        # Return results
        return arch_encoding + arch_features, target


def main():
    pass


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