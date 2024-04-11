from functools import partial
from collections import defaultdict
import random
import os

from PredictorDataGetter import DataGetter
from ArchitectureEncoder import *
from target_extrapolations import prefit_avg_initialization_extrapolation
from LearningCurveExtrapolator import *
from utils import ArchiSampler

from xgboost import XGBRegressor
from scipy.stats import spearmanr

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

if __name__ == '__main__':
    """
    """

    def add_feat_to_data(data: list[list[float]], feat: list[float]) -> list[list[float]]:
        data = np.column_stack((data,feat))
        return data
    
    dg = get_optimal_data_getter(OneHotOperation_Encoder())
    arch_i_sampler = ArchiSampler(dg.get_amount_of_architectures())

    ###
    ### Get boundaries for creation of artificial ZC proxies
    ###
    # Get max/min for normalization
    all_archs_i: list[int] = list(dg.data.keys())
    all_real_valaccs = [dg.get_real_val_acc_of_arch(arch_i) for arch_i in all_archs_i]
    min_vallacc, max_valacc = min(all_real_valaccs), max(all_real_valaccs)



    ###
    ### Define architectures used for training and testing         
    ###

    # Set architectures used for training and testing
    test_archs_i  = [arch_i_sampler.sample() for _ in range(5_000)]
    train_archs_i = [arch_i_sampler.sample() for _ in range(10_000)]
    
    ### Get training data
    train_inputs_clear = dg.get_arch_encoding_by_archs_i(train_archs_i)
    train_targets = np.array([dg.get_real_val_acc_of_arch(arch_i) for arch_i in train_archs_i])
    
    # Including the ZC proxies
    train_perfect_zc = np.array([(dg.get_real_val_acc_of_arch(arch_i) - min_vallacc) / (max_valacc - min_vallacc) for arch_i in train_archs_i])
    train_neg_perfect_zc = - train_perfect_zc
    train_random_zc = np.array([random.random() for _ in range(len(train_archs_i))])

    # Get testing data
    test_inputs_clear = dg.get_arch_encoding_by_archs_i(test_archs_i)
    test_targets = np.array([dg.get_real_val_acc_of_arch(arch_i) for arch_i in test_archs_i])
    
    # Including the ZC proxies
    test_perfect_zc = np.array([(dg.get_real_val_acc_of_arch(arch_i) - min_vallacc) / (max_valacc - min_vallacc) for arch_i in test_archs_i])
    test_neg_perfect_zc = - test_perfect_zc
    test_random_zc = np.array([random.random() for _ in range(len(test_archs_i))])

    
    print(f'Spearman on train set is: {spearmanr(train_random_zc, train_targets).statistic}')
    print(f'Spearman on test set is: {spearmanr(test_random_zc, test_targets).statistic}')
    
    ###
    ### Now that we have all of the data lets test the predictor
    ###

    def fit_and_measure_predictor(train_inputs, train_targets, test_inputs, test_targets) -> float:
        """
        Fits XGBRegressor on training data and returns the spearman of the predictions on test data
        """
        XGBoost_predictor = XGBRegressor()
        XGBoost_predictor.fit(train_inputs, train_targets)

        test_predictions = XGBoost_predictor.predict(test_inputs)
        return spearmanr(test_predictions, test_targets).statistic

    train_sets_sizes = [i for i in range(100, len(train_archs_i) + 1, 100)]
    data_results = defaultdict(list)

    # Try all 4 variations regarding usage of ZC proxies
    for i, train_set_size in enumerate(train_sets_sizes):
        print(f'Working on {i+1}/{len(train_sets_sizes)} train set size.')

        # Truncate the training data length according to `train_set_size`
        truncated_train_inputs_clear = train_inputs_clear[:train_set_size]
        truncated_train_targets = train_targets[:train_set_size]

        truncated_train_perfect_zc = train_perfect_zc[:train_set_size]
        truncated_train_neg_perfect_zc = train_neg_perfect_zc[:train_set_size]
        truncated_train_random_zc = train_random_zc[:train_set_size]


        # Start with NOT using any ZC
        spear_coef = fit_and_measure_predictor(truncated_train_inputs_clear, truncated_train_targets, test_inputs_clear, test_targets)
        data_results['Not using any ZC'].append(
            spear_coef
        )

        # Continue with using perfect ZC
        enhanced_train_inputs = add_feat_to_data(truncated_train_inputs_clear, truncated_train_perfect_zc)
        enhanced_test_inputs = add_feat_to_data(test_inputs_clear, test_perfect_zc)

        spear_coef = fit_and_measure_predictor(enhanced_train_inputs, truncated_train_targets, enhanced_test_inputs, test_targets)
        data_results['Using perfect ZC'].append(
             spear_coef
        )

        # Now continue with using neg perfect ZC
        enhanced_train_inputs = add_feat_to_data(truncated_train_inputs_clear, truncated_train_neg_perfect_zc)
        enhanced_test_inputs = add_feat_to_data(test_inputs_clear, test_neg_perfect_zc)

        spear_coef = fit_and_measure_predictor(enhanced_train_inputs, truncated_train_targets, enhanced_test_inputs, test_targets)
        data_results['Using neg-perfect ZC'].append(
             spear_coef
        )

        # And at the end get results with using random ZC
        enhanced_train_inputs = add_feat_to_data(truncated_train_inputs_clear, truncated_train_random_zc)
        enhanced_test_inputs = add_feat_to_data(test_inputs_clear, test_random_zc)

        spear_coef = fit_and_measure_predictor(enhanced_train_inputs, truncated_train_targets, enhanced_test_inputs, test_targets)
        data_results['Using random ZC'].append(
            spear_coef
        )

        
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 8))

    for label, data in data_results.items():
        ax.plot(train_sets_sizes, data, 'o--', label=label, alpha=0.5)

    ax.set_title(f'Dependence of the rank correlation on the size of the train set')
    ax.set_xlabel('Amount of architectures in the train set')
    ax.set_ylabel('Spearman rank correlation coefficient')
    fig.legend()

    plots_path = os.path.join('plot_thesis_figures', 'zc_hypothesis')
    
    os.makedirs(plots_path, exist_ok=True)
    fig.savefig(os.path.join(plots_path, f'Spearman_rank_corr_coef_vs_train_set_size_ZC_HYPOTHESIS.png'), dpi=300)
