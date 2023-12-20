import argparse
from enum import Enum
from collections import defaultdict
import random
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr 
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression

import normalizer
from best_features import BestFeatures

SUBJECTS = 47
PCA_N_COMPONENTS = 20
VOXELS_NUM = 91282
ALPHA_OPTIONS = np.concatenate((np.array([0.05, 0.1, 0.5, 1.0, 2.0, 5.0]),np.arange(50,900,5, dtype = 'float')))

SUBJECT_DATA_PATH = 'students_study/data/before/0{num_str}/MNINonLinear/Results/tfMRI_EmotionalNBack_AP/tfMRI_EmotionalNBack_AP_s4_MSMAll_hp2000_clean.feat/GrayordinatesStats/zstat1.dtseries.nii'
STUDENTS_SCORES = 'raven_scores_wo31.csv'

SEED = 191
random.seed(SEED)
np.random.seed(SEED)


# Initialize dictionaries to store chosen parameter values
chosen_values = defaultdict(lambda: defaultdict(int))

# Feature importance
important_features = np.zeros(VOXELS_NUM)

is_verbose = False

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Model(Enum):
    LINEAR_REGRESSION = 1
    RANDOM_FOREST = 2
    RIDGE = 3
    ELASTIC_NET = 4

MODELS = len(Model)

RANDOM_FOREST = "RF"
RANDOM_FOREST_SERIAL = Model.RANDOM_FOREST.value



REGRESSIONS = {
    "R": {      # Ridge
        "class": RidgeCV,
        "params": {
            "alphas": ALPHA_OPTIONS
        },
        "serial": Model.RIDGE.value,
        "name": "Ridge"

    },
    "EN": {     # ElasticNet
        "class": ElasticNetCV,
        "params": {
            "alphas": ALPHA_OPTIONS,
            "cv": 10,
            "max_iter": 10000
        },
        "serial": Model.ELASTIC_NET.value,
        "name": "Elastic Net"
    },
    "L": {      #Linear
        "class": LinearRegression,
        "params": {},
        "serial": Model.LINEAR_REGRESSION.value,
        "name": "Linear Regression"
        
    }

}
class ReduceDataMethod(Enum):
    PCA = (normalizer.NonNormalizer, 0)
    SUPERVISED = (normalizer.SKNormalizer, 1)
    UNSUPERVISED = (normalizer.SKNormalizer, 2)

    @classmethod
    def from_string(cls, method_string):
        for member in cls:
            if member.name.lower() == method_string.lower():
                return member
        raise ValueError(f"'{method_string}' is not a valid method")


def print_verbose(*args, **kwargs):
    if is_verbose:
        print(*args, **kwargs)

def read_fmri_data():
    fmri_data = []
    for i in range(1, SUBJECTS + 2):
        if i == 31: continue
        num_str = f"{i:02}"
        file_path = SUBJECT_DATA_PATH.format(num_str=num_str)
        fmri_img = nib.load(file_path)
        fmri_data.append(fmri_img.get_fdata()[0])
    return fmri_data

def print_models_statistics(scores, test_scores_pred, models_serials):
    for model in Model:
        if model.value not in models_serials:
            continue
        mse = np.mean((scores - test_scores_pred[:,(model.value-1)]) ** 2)
        print(f'{model} Mean squared error: {mse}')
        pearson = pearsonr(scores, test_scores_pred[:,(model.value-1)])
        print(f'{model} pearson: {pearson}')

def model_outcome_to_test_scores(test_scores_pred, test_idx, model_outcome, model):
        test_scores_pred[test_idx, (model - 1)] = np.round(model_outcome, 3)


def scores_predictions_models(args, scores, reduced_data, reduce_data_method, test_scores):
    loo = LeaveOneOut()
    test_scores_pred = np.zeros((SUBJECTS, MODELS))

    normalized_original_scores = np.zeros(scores.size)
    i = 1

    bfp = BestFeatures(0.8, 91200, True)
    bfn = BestFeatures(0.8, 91200, False)

    print_verbose(str(reduce_data_method))


    for train_idx, test_idx in loo.split(reduced_data):

        print_verbose("running subject number:", i)
        i += 1
        # Select the training and testing data
        fmri_train = np.array([reduced_data[idx] for idx in train_idx])
        scores_train = np.array([scores[idx] for idx in train_idx])
        fmri_test = np.array([reduced_data[idx] for idx in test_idx])

        # Normalize fmri data
        normalized = reduce_data_method.value[0](np.array(fmri_train))
        fmri_train = normalized.get_data_normalized()
        fmri_test = normalized.get_normalized_value(np.array(fmri_test))

        # Normalize scores data
        scores_data_normalizer = normalizer.ZNormalizer(scores_train)
        mean = np.mean(scores_train)
        std = np.std(scores_train)
        scores_train = scores_data_normalizer.get_data_normalized()
        test_score = test_scores[test_idx] if list(test_scores) else scores[test_idx]
        normalized_original_scores[test_idx] = scores_data_normalizer.get_normalized_value(test_score)

        # Preform feature selection methods
        match reduce_data_method:
            case ReduceDataMethod.PCA:
                pca = PCA(n_components=PCA_N_COMPONENTS)
                fit_fmri_train = pca.fit(fmri_train)
                fmri_train = pca.transform(fmri_train)
                bfp.calc_average_feature_importance(fit_fmri_train)
                bfn.calc_average_feature_importance(fit_fmri_train)
                print_verbose("\t>> real explained variance:", bfp.explained_var_real)
                fmri_test = pca.transform(fmri_test)
            case ReduceDataMethod.SUPERVISED:
                max_features = int(fmri_train.shape[1]* 0.1)
                selector = SelectFromModel(estimator = RandomForestRegressor(random_state=0, n_jobs=-1, criterion = "squared_error"), max_features = max_features)
                fmri_train = selector.fit_transform(fmri_train, scores_train)
                fmri_test = selector.transform(fmri_test)
            case ReduceDataMethod.UNSUPERVISED:
                max_features = int(fmri_train.shape[1]* 0.1)
                selector = SelectKBest(score_func=f_regression, k=max_features)
                fmri_train = selector.fit_transform(fmri_train, scores_train)
                fmri_test = selector.transform(fmri_test)
        
        models_serials = []
        # Random Forest regression model
        if args.regression.upper() == RANDOM_FOREST or args.regression.lower() == "all":
            print_verbose("\t>> running: Random Forest with Grid")
            param_grid = {
                'n_estimators': np.arange(50, 200, 50, dtype=int),  # Default value is 100
                'max_depth': np.append(np.arange(1, 20, 1, dtype=int), None),  # Default value is None
                'max_features': [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # Default value is 1.0
            }
            rf_model = RandomForestRegressor()
            grid_search = GridSearchCV(rf_model, param_grid, scoring=None, cv=10, n_jobs=-1)
            grid_search.fit(fmri_train, scores_train)
            rf_estimator = grid_search.best_estimator_
            rf_scores_pred = rf_estimator.predict(fmri_test)

            model_outcome_to_test_scores(test_scores_pred, test_idx, rf_scores_pred, RANDOM_FOREST_SERIAL)
            models_serials.append(RANDOM_FOREST_SERIAL)

        regressions = [args.regression] if args.regression.upper() in REGRESSIONS else [regression for regression in REGRESSIONS if args.regression == "all"]
        for regression in regressions:
            model_data = REGRESSIONS[regression]
            print_verbose("\t>> running:", model_data["name"])
            model = model_data["class"](**model_data["params"])
            model.fit(fmri_train, scores_train)
            model_scores_pred = model.predict(fmri_test)
            model_outcome_to_test_scores(test_scores_pred, test_idx, model_scores_pred, model_data["serial"])
            models_serials.append(model_data["serial"])

    # Compute the mean squared error between the predicted and actual test scores
    print_models_statistics(normalized_original_scores, test_scores_pred, models_serials)
    bfp.get_feature_importance(False)
    bfn.get_feature_importance(False)

    return test_scores_pred, normalized_original_scores

def runMainProgram(args, scores, fmri_data, test_scores=[]):
    if args.featureselection == 'all':
        data = []
        normalized = np.array(np.array([]))
        headers = ["LR + PCA", "LR + S_FS", "LR + US_FS", "RF + PCA", "RF + S_FS", "RF + US_FS", "RG + PCA", "RG + S_FS", "RG + US_FS", "EN + PCA", "EN + S_FS", "EN + US_FS", "normalized scores"]
        for reduce_method in ReduceDataMethod:
            returened_test_scores, normalized = scores_predictions_models(args, scores, fmri_data, reduce_method, test_scores)
            data.append(returened_test_scores)
        test_scores_pred = np.concatenate([*data, np.array([normalized]).transpose()], axis=1)
        np.savetxt(f'raw_data_{start_time}.csv', test_scores_pred, delimiter=",", header=",".join(headers), comments="")
        return test_scores_pred
    returned_test_scores, normalized = scores_predictions_models(args, scores, fmri_data, ReduceDataMethod.from_string(args.featureselection), test_scores)
    return returned_test_scores



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HI! \n Here is our model for predicting raven scores from brain fmri scans")
    parser.add_argument("--bootstrap", type=int, default=0, help="if the value is different from 0, it will run the program the given times with permutation in the raven scores")
    parser.add_argument("--regression", type=str, default="RF", help="RF for random forest \n L for linear regression \n R for ridge \n EN for elastic net \n all for all regression methods")
    parser.add_argument("--featureselection", "-fs", default="pca", type=str, help="choose feature selection method PCA | UNSUPERVISED | SUPERVISED | all")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()
    is_verbose = args.verbose


    bootstrap = f'run bootstraping for {args.bootstrap} rounds' if args.bootstrap > 0 else ''
    print_verbose("running the model with regression", args.regression, "and feature selection", args.featureselection, bootstrap)
    print(">> STARTING")
    print_verbose(">> Loading subjects' data")
    fmri_data = read_fmri_data()
    scores = []
    scores = np.loadtxt(STUDENTS_SCORES, 
                 max_rows=SUBJECTS, delimiter=",", usecols=[1])
    print_verbose(">> Done loading data")

    if args.bootstrap > 0:
        for i in range(args.bootstrap):
            print_verbose(">> Bootstraping raven scores values.")
            scores_shuffle = scores.copy()
            random.shuffle(scores_shuffle)
            runMainProgram(args, scores_shuffle, fmri_data, scores)
    else:
        test_scores_pred = runMainProgram(args, scores, fmri_data)

    print(">> DONE")

