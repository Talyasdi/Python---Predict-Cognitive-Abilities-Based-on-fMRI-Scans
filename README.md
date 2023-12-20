# final_project
final project in Ido Tavor's lab,
moderator: Maya Kadushin

# Raven Score Prediction Model

## Introduction

This is a Python script for predicting Raven scores from brain fMRI scans using various regression methods and feature selection techniques.

## Usage

To run the script, use the following command:

```bash
python3.10 model.py [options]
Options
--bootstrap: Specify the number of bootstrap iterations. If set to a value different from 0, the program will run the given number of times with permutation in the Raven scores.

--regression: Choose the regression method for prediction. Available options are:

RF: Random Forest
L: Linear Regression
R: Ridge Regression
EN: Elastic Net
all: Use all available regression methods

--featureselection or -fs: Choose the feature selection method. Available options are:

PCA: Principal Component Analysis
UNSUPERVISED: Unsupervised Feature Selection
SUPERVISED: Supervised Feature Selection
all: Use all available feature selection methods

--verbose or -v: Enable verbose mode to display detailed information during execution.

Example Usage
Here are some examples of how to use the script:

bash
# Run the program with Random Forest regression and PCA feature selection
python3.10 model.py --regression RF --featureselection PCA

# Run the program with Linear Regression, enabling verbose mode
python3.10 model.py --regression L --verbose

# Run the program with all regression methods and supervised feature selection
python model.py --regression all --featureselection SUPERVISED
# Run the program with Random Forest regression and PCA feature selection
python3.10 model.py --regression RF --featureselection PCA

# Run the program with Linear Regression, enabling verbose mode
python3.10 model.py --regression L --verbose

# Run the program with all regression methods and supervised feature selection
python3.10 model.py --regression all --featureselection SUPERVISED

## Notice!
this code requires python3.10 or a newer version. 



