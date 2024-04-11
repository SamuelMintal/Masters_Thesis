# Accompanying code for the Master Thesis at MFF CUNI named 'Leveraging lower fidelity proxies for neural network based NAS predictors'

This repository contains the code used for both generating the required data and the code of the proposed method and its experiments.

The base of the repository is given by code used for generating required data (namely adding ZC proxies scores into data from NAS-Bench 201) and saving them in .csv format. It has been forked from the https://github.com/SamsungLabs/zero-cost-nas and subsequently tailored for our needs. In order to generate the required data one firstly needs to run `nasbench2_pred.py`. This generates series of .csv files which then need to be merged by `___my_code/df_merger.py`. The last step is running the `___my_code/feature_engineering.ipynb` on the merged data.


The rest of the code (including previously mentioned `df_merger` and `feature_engineering.ipynb`) is written by us and is located in the `___my_code` folder. The code within this folder can further be split into 2 categories:

* The code responsible for the implementation of the method itself. Such code is located in files `ArchitectureEncoder.py`, `LearningCurveExtrapolator.py`, `PredictorDataGetter.py`, `target_extrapolations.py` and `utils.py`.

* The code responsible for running the experiments and thus producing plots displayed in the thesis. Depending on the type of the experiment it is located in one of the following files:
    * Extrapolation related experiments are in `Extrapolator_Experiments.py`
    * Predictor related experiments are in `PredictorModel_Experiments.py`
    * Performance of predictor depending on ZC proxies used is in `synthetic_ZC_proxies_experiments.py`

We also provide code for running Aging evolution NAS in the `main.py`
    


Author: Samuel Mintál