from time import time

import tensorflow as tf
import numpy as np

class LearningCurveExtrapolator:
    def __init__(self, name: str, func, var_list: list[tf.Variable], lr: float = 0.05) -> None:
        """
        Contructs Learning Curve Extrapolator for concrete function.

        Parameters
        ----------
        name : str
            Name of the function this extrapolator uses.

        func : lambda
            Function which this extrapolator uses. Should contain all variables from `var_list`.

        var_list : list[tf.Variable]
            list of tf.Variables which are in `func` and will be optimized during the fitting.

        lr : float, defualt=0.001
            Learning rate which the Adam optimizer will use during the fitting.
        """
        self.name = name

        self.func = func
        self.var_list = var_list

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    def get_name(self) -> str:
        """
        Returns the name of the function this extrapolator uses.
        """
        return self.name
    
    def fit_data(self, data: list[float], time_seconds: int = None, num_epochs: int = None, return_variables_progress: bool = False):
        """
        Fits the function to the supplied data.

        Parameters
        ----------
        data : list[float]
            the validation accuracies of the model. Note that at index i, we expect
            Validation accuracy after i-th learning epoch.

        time_seconds, num_epochs : int
            ALWAYS SPECIFIY ONLY ONE! 
            Selects until when the fitting will take place.
        """
        if return_variables_progress:
            progress_dict = {i: [] for i in range(len(self.var_list))}

        # If criterium is time
        if time_seconds is not None:
            start = time()
            # While we still have some time left
            while time() - start <= time_seconds:
                self._fit_data_epoch(data)
                if return_variables_progress:
                    self._append_variables_state(progress_dict)

        # Else if the criterium is number of epochs
        elif num_epochs is not None:
            # Train such amount of epochs
            for _ in range(num_epochs):
                self._fit_data_epoch(data)
                if return_variables_progress:
                    self._append_variables_state(progress_dict)

        # If neither variable was specified...
        else:
            raise ValueError('Specify one of time_seconds or num_epochs')

        if return_variables_progress:
            return progress_dict


    def _append_variables_state(self, var_dict: dict[str, list[float]]):
        """
        Appends variable state to dict with form dict[variable_index] == [1, 1.1, 1.2, 1.15, ...]
        """
        for i in range(len(self.var_list)):
            var_dict[i].append(self.var_list[i].numpy())

    def _fit_data_epoch(self, data: list[float]):
        """
        Fits data fro 1 epoch using MSE metric.
        """ 

        def MSE():
            loss = 0
            for i, data_sample in enumerate(data):
                loss += (self.func(i + 1) - data_sample) ** 2

            return loss / len(data)

        self.optimizer.minimize(MSE, var_list=self.var_list)

    
    def predict(self, epoch: int) -> float:
        """
        Predicts the validation accuracy from the validation accuracies curve supplied in fitting in supplied epoch.
        """
        return self.func(epoch)
    
class VaporPressure_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        a = tf.Variable(1.0)
        b = tf.Variable(1.0)
        c = tf.Variable(1.0)

        func = lambda x: np.e ** (a + (b / (x)) + c * np.log(x))

        super().__init__('VaporPressure_Extrapolator', func, [a, b, c])

class Pow3_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        c = tf.Variable(1.0)
        a = tf.Variable(1.0)
        alfa = tf.Variable(1.0)    

        func = lambda x: c - a * (x ** (- alfa))

        super().__init__('Pow3_Extrapolator', func, [a, alfa, c])

class LogLogLinear_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        a = tf.Variable(1.0)
        b = tf.Variable(1.0)
        
        func = lambda x: tf.math.log(a * np.log(x) + b)

        super().__init__('LogLogLinear_Extrapolator', func, [a, b], lr=10)

class LogPower(LearningCurveExtrapolator):
    def __init__(self) -> None:
        a = tf.Variable(1.0)
        b = tf.Variable(1.0)
        c = tf.Variable(1.0)

        func = lambda x: a / (1 + ((x / (np.e ** b)) ** c))

        super().__init__('LogPower_Extrapolator', func, [a, b, c])

class Pow4_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        c = tf.Variable(1.0)
        a = tf.Variable(1.0)
        b = tf.Variable(1.0)
        alfa = tf.Variable(1.0)    

        func = lambda x: c - ((a * x + b) ** (- alfa))

        super().__init__('Pow4_Extrapolator', func, [c, a, b, alfa], lr=10)

class MMF_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        alfa = tf.Variable(1.0)    
        beta = tf.Variable(1.0)   
        delta = tf.Variable(1.0)
        k = tf.Variable(1.0)   

        func = lambda x: alfa - ((alfa - beta) / (1 + ((k * x) ** delta)))

        super().__init__('MMF_Extrapolator', func, [alfa, beta, delta, k])

class Exp4_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        c = tf.Variable(1.0)
        a = tf.Variable(1.0)
        b = tf.Variable(1.0)
        alfa = tf.Variable(1.0)    

        func = lambda x: c - (np.e ** (- a * (x ** alfa) + b))

        super().__init__('Exp4_Extrapolator', func, [c, a, b, alfa])


class Janoschek_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        alfa = tf.Variable(1.0)
        beta = tf.Variable(1.0)
        k = tf.Variable(1.0)
        delta = tf.Variable(1.0)    

        func = lambda x: alfa - (alfa - beta) * (np.e ** (- k * (x ** delta)))

        super().__init__('Janoschek_Extrapolator', func, [beta, k, delta, alfa])

class Weibull_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        alfa = tf.Variable(1.0)
        beta = tf.Variable(1.0)
        k = tf.Variable(1.0)
        delta = tf.Variable(1.0)    

        func = lambda x: alfa - (alfa - beta) * (np.e ** (- ((k * x) ** delta)))

        super().__init__('Weibull_Extrapolator', func, [beta, k, delta, alfa])

class Ilog2_Extrapolator(LearningCurveExtrapolator):
    def __init__(self) -> None:

        a = tf.Variable(1.0)
        c = tf.Variable(1.0)

        func = lambda x: c - (a / np.log(x))

        super().__init__('Ilog2_Extrapolator', func, [a, c])


##############################################
### Learning Curve Extrapolators Ensembler ###
##############################################

class LearningCurveExtrapolatorsEnsembler:
    def __init__(self, extrapolators: list[LearningCurveExtrapolator], verbose: bool = True) -> None:
        self.extrapolators: list[LearningCurveExtrapolator] = extrapolators
        self.verbose: bool = verbose

    def fit_extrapolators(self, data: list[float], time_seconds: int = None, num_epochs: int = None) -> None:
        """
        Fits all the extrapolators
        """

        for extrapolator in self.extrapolators:
            if self.verbose:
                print(f'Fitting extrapolator {extrapolator.get_name()}')

            if time_seconds is not None:
                extrapolator.fit_data(data, time_seconds=time_seconds)
            elif num_epochs is not None:
                extrapolator.fit_data(data, num_epochs=num_epochs)
            else:
                raise ValueError('Specify one of time_seconds or num_epochs')


    def predict_avg(self, epoch: int):
        """
        Returns average of all learning curve extrapolators predictions
        """
        return sum([ex.predict(epoch) for ex in self.extrapolators]) / len(self.extrapolators)
    

    def plot_data(self, data: list[float], trained_on_n_samples: int):
        epochs = np.array(range(len(data))) + 1

        plt.plot(epochs, val_accs, label='Ground truth')
        plt.plot(epochs, [self.predict_avg(i) for i in epochs], label='Ensembler average')

        for lce in self.extrapolators:
            plt.plot(epochs, [lce.predict(i) for i in epochs], label=lce.get_name())
            
        plt.axvline(trained_on_n_samples, color='red')
        plt.legend()
        plt.show()


def test_ensembler(data, lcee: LearningCurveExtrapolatorsEnsembler, train_on_first: int = 20):
    train_data = data[:train_on_first]
    epochs = np.array(range(len(data))) + 1

    lcee.fit_extrapolators(train_data, time_seconds=60)

    plt.plot(epochs, data, label='Ground truth')
    plt.plot(epochs, [lcee.predict_avg(i) for i in epochs], label='Ensembler average')

    for lce in lcee.extrapolators:
        plt.plot(epochs, [lce.predict(i) for i in epochs], label=lce.get_name())
    
    plt.axvline(train_on_first, color='red')
    plt.legend()
    plt.show()

def test_extrapolator(data: list[float], extrapolator: LearningCurveExtrapolator, train_on_first: int = 20):
    train_data = data[:train_on_first]
    epochs = np.array(range(len(data))) + 1

    variables_progress = extrapolator.fit_data(train_data, time_seconds=60, return_variables_progress=True)
    
    # Plot how the predicted curve compares to the original
    plt.plot(epochs, data, label='Ground truth')
    plt.plot(epochs, [extrapolator.predict(i) for i in epochs], label=extrapolator.get_name())

    plt.axvline(train_on_first, color='red')
    plt.legend()
    plt.show()


    # Plot variables progress
    for k, v in variables_progress.items():
        training_epochs = np.array(list(range(len(v)))) + 1
        plt.plot(training_epochs, v, label=f'Progress of variable {k}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    from ast import literal_eval
    import os

    import pandas as pd
    import matplotlib.pyplot as plt

    CLEAN_DATA_PATH = os.path.join('___my_code', '_clean_all_merged.csv')
    CONVERT_TO_LIST_COLUMNS = ['train_accs', 'val_accs']
    ARCH_INDEX = 1

    # Load data
    data_df = pd.read_csv(CLEAN_DATA_PATH, index_col='arch_i')
    for conv_to_list_col in CONVERT_TO_LIST_COLUMNS:
        data_df[conv_to_list_col] = data_df[conv_to_list_col].apply(literal_eval)
    data_dict = data_df.to_dict('index')

    # Get indexed arch's val accs
    arch_i_data = data_dict[ARCH_INDEX]
    val_accs = arch_i_data['val_accs']


    test_extrapolator(val_accs, Pow4_Extrapolator())
    #exit()
    test_ensembler(
        val_accs, 
        LearningCurveExtrapolatorsEnsembler(
            [
                VaporPressure_Extrapolator(), 
                Pow3_Extrapolator(), 
                LogLogLinear_Extrapolator(),
                LogPower(),
                Pow4_Extrapolator(),
                MMF_Extrapolator(),
                Exp4_Extrapolator(),
                Janoschek_Extrapolator(),
                Weibull_Extrapolator(),
                Ilog2_Extrapolator()
            ]
        )
    )