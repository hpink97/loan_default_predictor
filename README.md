# Loan Default Prediction

This repository contains code and data for predicting loan defaulting. The goal is to build a machine learning model that can accurately classify borrowers as either likely to default on their loans or not.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Aggregation and Feature Engineering](#aggregation-and-feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Results](#results)

## Introduction

The objective of this project is to develop a predictive model that can assess the credit risk of borrowers and predict loan defaulting. By analyzing historical loan data and utilizing machine learning techniques, we aim to build a robust model that can assist in making informed decisions regarding loan approvals and risk management.

## Data

The project utilizes a dataset containing various features related to borrowers and loan applications. The dataset includes information such as borrower demographics, financial indicators, credit history, and loan characteristics. The data is obtained from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)].

## Aggregation and Feature Engineering

The aggregation and feature engineering steps are performed in the `01_credit_risk_eda.Rmd` file. This script combines data from multiple sources and performs preprocessing tasks such as:

- Importing and cleaning the application data (`application_train.csv` and `application_test.csv`).
- Creating additional features based on the existing data, such as income ratios and discretizing continuous features.
- Importing and preprocessing additional datasets, such as bureau data, credit card data, installment payment data, and previous credit application data.
- Aggregating the data at the loan applicant level (`sk_id_curr`) by calculating various statistics and ratios.
- Handling missing values by replacing them with zeros in the aggregated columns.

These steps aim to transform the raw data into a merged and aggregated dataset suitable for model training and evaluation.

For more details, please refer to the [`01_credit_risk_eda.Rmd`](01_credit_risk_eda.Rmd) file.

## Model Training

The project utilizes the XGBoost algorithm for training the loan default prediction model. The [`Model`](03_ml_preprocessing_and_training.ipynb) class defined in the `03_ml_preprocessing_and_training.ipynb` notebook provides a streamlined workflow for feature selection, model training, and evaluation. The steps involved in model training are as follows:

1. Preprocessing: The dataset is preprocessed to handle missing values, impute data, scale numeric columns, and encode categorical variables.
2. Splitting Data: The preprocessed dataset is split into training, evaluation, and testing sets.
3. Feature Selection: The model selects the top `n` features using the `SelectKBest` algorithm based on mutual information classification scores.
4. Model Training: The XGBoost classifier is trained using the selected features and specified hyperparameters.
5. Model Evaluation: The trained model's performance is evaluated using various metrics such as F1 score, accuracy, precision, recall, specificity, and ROC AUC score.

### Dataset class for ML preprocessing

The `Dataset` class is designed to handle the preprocessing and splitting of a dataset for machine learning tasks. Here is a summary of its functionality:

1. Initialization: The class takes in a pandas DataFrame (`df`) representing the dataset and the target variable (`target`). It also accepts additional parameters like `is_test` to indicate if the dataset is a test set, `scaler` for scaling numeric columns, and `trained_cols` for indicating specific columns to be used in the final dataset.

2. Preprocessing: The `preprocess()` method performs preprocessing tasks on the dataset. It includes basic imputations for missing values, smart imputations using the `miceforest` package for remaining missing values, scaling of numeric columns, and label encoding for binary columns. It also performs one-hot encoding for categorical columns.

3. Splitting Data: The `split_data()` method splits the preprocessed dataset into training, evaluation, and testing sets. It takes parameters like `test_size` and `eval_size` to control the size of the test and evaluation sets, respectively. It prints information about the sizes and positive rates of each split.


### `Model` Class for training xgboost models

The model class provides a streamlined workflow for feature selection, model training, and evaluation of xgboost models.

Here is a summary of the key features and methods of the Model class:

1.   **Initialisation**: The class is initialized with the necessary input data using the `Dataset` class, including the training and test sets (`X_train`, `X_test`) and their corresponding target variables (`y_train`, `y_test`).
2.   **Feature Selection**: The `select_features()` method allows you to perform feature selection using the `sklearn.feature_selection.SelectKBest` algorithm. It selects the top num_features based on mutual information classification scores.
3. **Model Training**: The `train_model()` method trains an XGBoost classifier using the specified xgboost_params. It uses the training data and evaluates the model's performance on the evaluation set (`X_eval`, `y_eval`). Early stopping is implemented to prevent overfitting.
4. **Model Evaluation**: The `evaluate_model()` method calculates and prints various evaluation metrics, including F1 score, accuracy, precision, recall, specificity, ROC AUC score, and balanced accuracy. It also selects the optimal threshold for determining binary predictions based on the F1 score.
5. **Performance Visualisation**: The class provides several plotting methods to visualise model performance, including `plot_roc_auc()` to visualize the ROC curve and calculate the AUC score, `plot_predictions()` to plot the predicted probabilities against the true labels, and `plot_feature_importance()` to display the feature importances using a bar plot.

## Results

