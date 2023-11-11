from time import time

import tensorflow as tf
import numpy as np

from utils import load_data

class LearningCurveExtrapolator:
    def __init__(self, name: str, func, var_list: list[tf.Variable], var_names: list[str], lr: float = 0.05) -> None:
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

        var_names : list[str]
            names of the variables in the `var_list` (Must be index matched)

        lr : float, defualt=0.001
            Learning rate which the Adam optimizer will use during the fitting.
        """
        self.name = name

        self.func = func
        self.var_list = var_list
        self.var_names = var_names

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    def get_name(self) -> str:
        """
        Returns the name of the function this extrapolator uses.
        """
        return self.name
    
    def _mean_abs_error(self, predictions: list[float], targets: list[float]) -> float:
        """
        Returns mean absolute error given predictions and targets
        """
        assert len(predictions) == len(targets)
        mae = 0

        for pred, targ in zip(predictions, targets):
            mae += abs(pred - targ)

        return mae / len(predictions)
        


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

        return_variables_progress : bool
            If True functions returns dict[str, list[float]] of variables values progress and MAE curve of the fit
        """
        if return_variables_progress:
            progress_dict = {var_name: [] for var_name in self.var_names}
            mae_progress = []

        # If criterium is time
        if time_seconds is not None:
            start = time()
            # While we still have some time left
            while time() - start <= time_seconds:
                self._fit_data_epoch(data)
                if return_variables_progress:
                    self._append_variables_state(progress_dict)
                    mae_progress.append(self._mean_abs_error([self.predict(e) for e in range(1, len(data) + 1)], data))

        # Else if the criterium is number of epochs
        elif num_epochs is not None:
            # Train such amount of epochs
            for _ in range(num_epochs):
                self._fit_data_epoch(data)
                if return_variables_progress:
                    self._append_variables_state(progress_dict)
                    mae_progress.append(self._mean_abs_error([self.predict(e) for e in range(1, len(data) + 1)], data))

        # If neither variable was specified...
        else:
            raise ValueError('Specify one of time_seconds or num_epochs')

        if return_variables_progress:
            return progress_dict, mae_progress


    def _append_variables_state(self, var_dict: dict[str, list[float]]):
        """
        Appends variable state to dict with form dict[variable_index] == [1, 1.1, 1.2, 1.15, ...]
        """
        for i, var_name in enumerate(self.var_names):
            var_dict[var_name].append(self.var_list[i].numpy())

    def _fit_data_epoch(self, data: list[float]):
        """
        Fits data for 1 epoch using MSE metric.
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
    def __init__(
            self, 
            a: float = 1.0, 
            b: float = 1.0, 
            c: float = 1.0
        ) -> None:

        a = tf.Variable(a)
        b = tf.Variable(b)
        c = tf.Variable(c)

        func = lambda x: np.e ** (a + (b / (x)) + c * np.log(x))

        super().__init__('VaporPressure_Extrapolator', func, [a, b, c], ['a', 'b', 'c'])

class Pow3_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            a: float = 1.0,
            c: float = 1.0,
            alfa: float = 1.0
        ) -> None:

        c = tf.Variable(c)
        a = tf.Variable(a)
        alfa = tf.Variable(alfa)    

        func = lambda x: c - a * (x ** (- alfa))

        super().__init__('Pow3_Extrapolator', func, [a, alfa, c], ['a', 'alfa', 'c'])

class LogLogLinear_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            a: float = 1.0,
            b: float = 1.0
        ) -> None:

        a = tf.Variable(a)
        b = tf.Variable(b)
        
        func = lambda x: tf.math.log(a * np.log(x) + b)

        super().__init__('LogLogLinear_Extrapolator', func, [a, b], ['a', 'b'])

class LogPower(LearningCurveExtrapolator):
    def __init__(
            self,
            a: float = 1.0,
            b: float = 1.0,
            c: float = 1.0
        ) -> None:
        a = tf.Variable(a)
        b = tf.Variable(b)
        c = tf.Variable(c)

        func = lambda x: a / (1 + ((x / (np.e ** b)) ** c))

        super().__init__('LogPower_Extrapolator', func, [a, b, c], ['a', 'b', 'c'])

class Pow4_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            c: float = 1.0,
            a: float = 1.0,
            b: float = 1.0,
            alfa: float = 1.0
        ) -> None:

        c = tf.Variable(c)
        a = tf.Variable(a)
        b = tf.Variable(b)
        alfa = tf.Variable(alfa)    

        func = lambda x: c - ((a * x + b) ** (- alfa))

        super().__init__('Pow4_Extrapolator', func, [c, a, b, alfa], ['c', 'a', 'b', 'alfa'])

class MMF_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            alfa: float = 1.0,
            beta: float = 1.0,
            delta: float = 1.0,
            k: float = 1.0
        ) -> None:

        alfa = tf.Variable(alfa)    
        beta = tf.Variable(beta)   
        delta = tf.Variable(delta)
        k = tf.Variable(k)   

        func = lambda x: alfa - ((alfa - beta) / (1 + ((k * x) ** delta)))

        super().__init__('MMF_Extrapolator', func, [alfa, beta, delta, k], ['alfa', 'beta', 'delta', 'k'])

class Exp4_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            c: float = 70.0,
            a: float = 1.0,
            b: float = 1.0,
            alfa: float = 1.0
        ) -> None:

        c = tf.Variable(c)
        a = tf.Variable(a)
        b = tf.Variable(b)
        alfa = tf.Variable(alfa)    

        func = lambda x: c - (np.e ** (- a * (x ** alfa) + b))

        super().__init__('Exp4_Extrapolator', func, [c, a, b, alfa], ['c', 'a', 'b', 'alfa'])


class Janoschek_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            alfa: float = 1.0,
            beta: float = 1.0,
            k: float = 1.0,
            delta: float = 1.0
        ) -> None:

        alfa = tf.Variable(alfa)
        beta = tf.Variable(beta)
        k = tf.Variable(k)
        delta = tf.Variable(delta)    

        func = lambda x: alfa - (alfa - beta) * (np.e ** (- k * (x ** delta)))

        super().__init__('Janoschek_Extrapolator', func, [beta, k, delta, alfa], ['beta', 'k', 'delta', 'alfa'])

class Weibull_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            alfa: float = 1.0,
            beta: float = 1.0,
            k: float = 1.0,
            delta: float = 1.0
        ) -> None:

        alfa = tf.Variable(alfa)
        beta = tf.Variable(beta)
        k = tf.Variable(k)
        delta = tf.Variable(delta)    

        func = lambda x: alfa - (alfa - beta) * (np.e ** (- ((k * x) ** delta)))

        super().__init__('Weibull_Extrapolator', func, [beta, k, delta, alfa], ['beta', 'k', 'delta', 'alfa'])

class Ilog2_Extrapolator(LearningCurveExtrapolator):
    def __init__(
            self,
            a: float = 1.0,
            c: float = 1.0
        ) -> None:

        a = tf.Variable(a)
        c = tf.Variable(c)

        func = lambda x: c - (a / np.log(x))

        super().__init__('Ilog2_Extrapolator', func, [a, c], ['a', 'c'])

##########################
### Mine extrapolators ###
##########################

class LogShiftScale_Extrapolator(LearningCurveExtrapolator):
    """
    is equivalent to: scale * log(inside_scale * x + inside_shift) + shift
    """
    def __init__(
            self,
            scale: float = 1.0,
            shift: float = 0.0,

            inside_scale: float = 1.0,
            inside_shift: float = 0.0
        ) -> None:

        scale = tf.Variable(scale)
        shift = tf.Variable(shift)

        inside_scale = tf.Variable(inside_scale)
        inside_shift = tf.Variable(inside_shift)

        func = lambda x: scale * tf.math.log(inside_scale * x + inside_shift) + shift

        super().__init__('LogShiftScale_Extrapolator', func, [scale, shift, inside_scale, inside_shift], ['scale', 'shift', 'inside_scale', 'inside_shift'])

class LogAllFree_Extrapolator(LearningCurveExtrapolator):
    """
    After succes of LogShiftScale_Extrapolator, we decided to make it even more free with addition of:
        1. (kx) ** power instead of having only x
        2. dividing it whole by (division_scale * (x ** division_pow) + divison_shift)

    is equivalent to: (scale * log(inside_scale * (k_inside * x) ** inside_pow + inside_shift) + shift) / (division_scale * (x ** division_pow) + divison_shift)
    """
    def __init__(
            self,
            scale: float = 1.0,
            shift: float = 0.0,

            inside_scale: float = 1.0,
            inside_shift: float = 0.0,
            inside_pow: float = 1.0,
            inside_k: float = 1.0,

            division_scale: float = 0.0,
            division_pow: float = 0.0,
            divison_shift: float = 1.0,
        ) -> None:

        scale = tf.Variable(scale)
        shift = tf.Variable(shift)

        inside_scale = tf.Variable(inside_scale)
        inside_shift = tf.Variable(inside_shift)
        inside_pow = tf.Variable(inside_pow)
        inside_k = tf.Variable(inside_k)

        division_scale = tf.Variable(division_scale)
        division_pow   = tf.Variable(division_pow)
        divison_shift  = tf.Variable(divison_shift)

        func = lambda x: (scale * tf.math.log(inside_scale * ((inside_k * x) ** inside_pow) + inside_shift) + shift) / (division_scale * (x ** division_pow) + divison_shift)

        super().__init__('LogAllFree_Extrapolator', func, [scale, shift, inside_scale, inside_shift, inside_pow, inside_k, division_scale, division_pow, divison_shift], ['scale', 'shift', 'inside_scale', 'inside_shift', 'inside_pow', 'inside_k', 'division_scale', 'division_pow', 'divison_shift'])

##############################################
### Learning Curve Extrapolators Ensembler ###
##############################################

class LearningCurveExtrapolatorsEnsembler:
    def __init__(self, extrapolators: list[LearningCurveExtrapolator], verbose: bool = True) -> None:
        self.extrapolators: list[LearningCurveExtrapolator] = extrapolators
        self.verbose: bool = verbose

    def show_extrapolators_list(self) -> list[LearningCurveExtrapolator]:
        """
        Returns list of lc extrapolators this Ensembler uses
        """
        return self.extrapolators

    def fit_extrapolators(self, data: list[float], time_seconds: int = None, num_epochs: int = None) -> None:
        """
        Fits all the extrapolators each for `time_seconds` seconds
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


    def predict_avg(self, epoch: int) -> float:
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

    for lce in lcee.extrapolators:
        plt.plot(epochs, [lce.predict(i) for i in epochs], label=lce.get_name())

    plt.plot(epochs, data, label='Ground truth')
    plt.plot(epochs, [lcee.predict_avg(i) for i in epochs], label='Ensembler average', color='black')


    plt.axvline(train_on_first, color='black')
    plt.legend()
    plt.show()


def test_extrapolator(data: list[float], extrapolator: LearningCurveExtrapolator, train_on_first: int = 20, time_seconds: int = 60):
    train_data = data[:train_on_first]
    epochs = np.array(range(len(data))) + 1

    variables_progress, mae_progress = extrapolator.fit_data(train_data, time_seconds=time_seconds, return_variables_progress=True)
    training_epochs = np.array(list(range(len(mae_progress)))) + 1
    
    fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(20, 5))



    # Plot how the predicted curve compares to the original
    axes[0].plot(epochs, data, label='Ground truth')
    axes[0].plot(epochs, [extrapolator.predict(i) for i in epochs], label=extrapolator.get_name())

    axes[0].axvline(train_on_first, color='black')

    axes[0].set_title(f'Extrapolator {extrapolator.get_name()} performance against target curve')
    axes[0].set_ylabel('Validation accuracy [%]')
    axes[0].set_xlabel('Predicted epoch')
    axes[0].legend()


    # Plot variables progress
    for k, v in variables_progress.items():
        axes[1].plot(training_epochs, v, label=f'Progress of variable {k}')

    axes[1].set_title('Variable values throughout training epochs')
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Training epoch')
    axes[1].legend()



    # Plot MAE progress
    axes[2].plot(training_epochs, mae_progress)
    axes[2].set_title('MAE progress throughout training epochs')
    axes[2].set_ylabel('MAE')
    axes[2].set_xlabel('Training epoch')

    plt.savefig(os.path.join('Extrapolators_curves', f'curves_of_{extrapolator.get_name()}'))
    #plt.show()


if __name__ == '__main__':
    """

    The question in pretraining the extrapolators.

    To how long part is given as training data, as it's shaped oposed to the rest of the learning curve
    Can change which extrapolators are worse or better due to theyr inherit drop off/exponential growth characteristics....

    """
    import os
    import matplotlib.pyplot as plt

    data_dict = load_data()
    ARCH_INDEX = 0
    
    # Get indexed arch's val accs
    arch_i_data = data_dict[ARCH_INDEX]
    val_accs = arch_i_data['val_accs']

    def get_fresh_extrapolators() -> list[LearningCurveExtrapolator]:
        return [
            VaporPressure_Extrapolator(), 
            Pow3_Extrapolator(), 
            LogLogLinear_Extrapolator(),
            LogPower(),
            Pow4_Extrapolator(),
            MMF_Extrapolator(),
            Exp4_Extrapolator(),
            Janoschek_Extrapolator(),
            Weibull_Extrapolator(),
            Ilog2_Extrapolator(),
            LogShiftScale_Extrapolator(),
            LogAllFree_Extrapolator()
        ]

    extrapolators_to_test = get_fresh_extrapolators()
    for extrapolator in extrapolators_to_test:
        test_extrapolator(val_accs, extrapolator, time_seconds=600)
        print(f'Done {extrapolator.get_name()}')
    
    #exit()
    """
    test_ensembler(
        val_accs, 
        LearningCurveExtrapolatorsEnsembler(
            get_fresh_extrapolators()
        )
    )
    #"""