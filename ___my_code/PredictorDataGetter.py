from utils import *
from ArchitectureEncoder import *
from LearningCurveExtrapolator import *

class DataGetter:
    """
    Class providing abstraction over data.
    """
    def __init__(self, arch_encoder: ArchitectureEncoder, features_to_get: list[str], data_path: str = os.path.join('___my_code', '_clean_all_merged.csv')) -> None:
        """
        Loads data into memory and sets ArchitectureEncoder and feature list which define what the input for the perforamnce predictor will be.

        Parameters
        ----------
        arch_encoder : ArchitectureEncoder
            Architecture encoder which to use for encoding the architectures.

        features_to_get : list[str]
            list of features which to retrieve for prediction purposes.
        """
        self.arch_encoder = arch_encoder
        self.features_to_get = features_to_get
        self.data = load_data(data_path)

        # Maps string-ified standart encoding of an architecture to it's arch_i (by which we index `self.data`)
        self.std_encoding_to_arch_i = {
            str(self.arch_encoder.arch_str_to_standart_encoding(arch_data['arch_str'])): arch_i for (arch_i, arch_data) in self.data.items()
        }

    def get_arch_i_from_standart_encoding(self, std_encoding: list[int]) -> int:
        """
        Returns arch_i of architecture which encodes to `std_encoding`
        """
        return self.std_encoding_to_arch_i[str(std_encoding)]

    def get_std_encoding_of_arch_i(self, arch_i: int) -> list[int]:
        """
        Returns standart encoding of architecture specified by it's NasBench201 index `arch_i`.
        """
        return self.arch_encoder.arch_str_to_standart_encoding(self.data[arch_i]['arch_str'])

    def get_perf_predictor_input_length(self) -> int:
        """
        Returns the length of the input for the performance predictor which
        depends on the current ArchitectureEncoder and features list set in the class.
        """
        return len(self.features_to_get) + self.arch_encoder.get_encoding_length()

    def get_arch_indices(self) -> list[int]:
        """
        Returns all indices of available architectures
        """
        return list(self.data.keys())
    
    def get_real_val_acc_of_arch(self, arch_i: int) -> float:
        """
        Returns final (after 200th training epoch) validational 
        accuracy of architecture specified by `arch_i`.
        """
        return self.data[arch_i]['val_accs'][-1]
    
    def get_amount_of_architectures(self) -> int:
        """
        Returns amount of architectures in the loaded data.
        """
        return len(self.data)        

    def get_prediction_features_by_arch_index(self, arch_index: int) ->list[float]:
        """
        Returns input features used for prediction by performance predictor.

        Parameters
        ----------
        arch_index : int 
            Index of an architecture for which to retrieve the features (arch_i).

        Returns
        -------
        list[float]:
            features of architecture for prediction
        """
        # Firstly get architecture's data
        arch_data = self.data[arch_index]
        
        # Secondly get architecture's encoding
        arch_encoding = self.arch_encoder.convert_from_arch_string(arch_data['arch_str'])

        # Thirdly collect the required features
        arch_features = [arch_data[feat] for feat in self.features_to_get]

        return arch_encoding + arch_features
        

    def get_training_data_by_arch_index(
            self,
            arch_index: int, 
            lc_extrapolaion_info: tuple[LearningCurveExtrapolatorsEnsembler, int, int]
        ) -> tuple[list[int], int]:
        """
        Returns input features as well as extrapolated target for specified architecture.

        Parameters
        ----------
        arch_index : int 
            Index of an architecture for which to retrieve the data (arch_i).        

        lc_extrapolaion_info : tuple[LearningCurveExtrapolatorsEnsembler, int, int]
            Tuple of 3 elements containing all of the information about target generation:
            - LearningCurveExtrapolatorsEnsembler used for extrapolating target.            
            - Amount of architecture's training epochs on which the extrapolator will be fitted.
            - The amount of seconds for which each extrapolator in ensembler will be able to do it's fitting.

        Returns
        -------
        tuple[list[int], int]
            tuple containing training-data and it's target
        """
        # Firstly get input features
        input_features = self.get_prediction_features_by_arch_index(arch_index)

        # And Secondly before we return results, get the target
        # validation accuracy of the architecture
        lc_extrapolator, n_training_data, n_training_secs = lc_extrapolaion_info
        trainning_data = self.data[arch_index]['val_accs'][:n_training_data]
        lc_extrapolator.fit_extrapolators(trainning_data, time_seconds=n_training_secs)
        target = lc_extrapolator.predict_avg(200).numpy()

        # Return results
        return input_features, target