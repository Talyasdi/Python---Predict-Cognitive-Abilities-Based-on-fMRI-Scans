# Predict Cognitive Abilities Based on fMRI Scans README
Final project in Ido Tavor's lab

## Introduction
This repository contains code for predicting Raven scores from brain fMRI scans. The prediction is performed using various regression models, including Random Forest, Linear Regression, Ridge Regression, and Elastic Net. Additionally, different feature selection methods are employed, such as Principal Component Analysis (PCA), Unsupervised Feature Selection (US_FS), and Supervised Feature Selection (S_FS).

## Files
1. **best_features.py**: This file defines a class `BestFeatures` responsible for tracking and calculating the best features based on the explained variance and maximum features.

2. **model.py**: The `model.py` file contains the main logic for running the prediction models. It supports different regression models and feature selection methods. The code is modular and allows for easy extension with additional models.

3. **normalizer.py**: This file includes classes for data normalization. The normalization methods provided are Standard Scaler, Min-Max Normalizer, Z-Score Normalizer, and Non-Normalizer.

## Running the Code
### Running the Prediction
To run the prediction, execute the following command for example:
```bash
python model.py --bootstrap 0 --regression RF --featureselection pca --verbose
```
- `--bootstrap`: Number of bootstrap rounds. Set to 0 for a regular run.
- `--regression`: Specify the regression model (RF, L, R, EN, all).
- `--featureselection`: Choose feature selection method (pca, unsupervised, supervised, all).
- `--verbose`: Enable verbose mode for detailed output.

## Output
The code generates output files containing the predicted scores and feature importance values. Output files are saved in the project directory.

- Feature Importance files: These files contain the importance of each feature after each round.
- Model Outcome file: Contains the predicted scores for each subject using different regression models and feature selection methods.

## Conclusion
This code provides a flexible framework for predicting Raven scores from brain fMRI scans. Experiment with different configurations to find the best-performing model for your specific dataset.

## Notice!
This code requires python3.10 or a newer version. 



