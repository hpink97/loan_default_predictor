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
- [Contributing](#contributing)
- [License](#license)

## Introduction

The objective of this project is to develop a predictive model that can assess the credit risk of borrowers and predict loan defaulting. By analyzing historical loan data and utilizing machine learning techniques, we aim to build a robust model that can assist in making informed decisions regarding loan approvals and risk management.

## Data

The project utilizes a dataset containing various features related to borrowers and loan applications. The dataset includes information such as borrower demographics, financial indicators, credit history, and loan characteristics. The data is obtained from [source name or link].

## Aggregation and Feature Engineering

The feature engineering and aggregation steps are performed in the [`01_credit_risk_aggregate_and_merge.Rmd`](01_credit_risk_aggregate_and_merge.Rmd) file. This script combines data from multiple sources and performs preprocessing tasks such as:

- Aggregating information from different datasets to create new features.
- Handling missing values and imputing them using various strategies.
- Scaling numeric columns to ensure feature comparability.
- Encoding categorical variables for model compatibility.
- Selecting relevant features based on domain knowledge and analysis.

These steps aim to transform the raw data into a suitable format for model training and improve the predictive power of the features.

## Model Training

The project utilizes the XGBoost algorithm for training the loan default prediction model. The [`Model`](03_ml_preprocessing_and_training.ipynb) class defined in the `03_ml_preprocessing_and_training.ipynb` notebook provides a streamlined workflow for feature selection, model training, and evaluation. The steps involved in model training are as follows:

1. Preprocessing: The dataset is preprocessed to handle missing values, impute data, scale numeric columns, and encode categorical variables.
2. Splitting Data: The preprocessed dataset is split into training, evaluation, and testing sets.
3. Feature Selection: The model selects the top `n` features using the `SelectKBest` algorithm based on mutual information classification scores.
4. Model Training: The XGBoost classifier is trained using the selected features and specified hyperparameters.
5. Model Evaluation: The trained model's performance is evaluated using various metrics such as F1 score, accuracy, precision, recall, specificity, and ROC AUC score.

## Model Evaluation

The performance of the loan default prediction model is evaluated using various evaluation metrics to assess its accuracy and effectiveness. The [`Model`](03_ml_preprocessing_and_training.ipynb) class provides methods for evaluating the model and visualizing its performance, including:

- F1 score: The harmonic mean of precision and recall, which balances both metrics.
- Accuracy: The overall accuracy of the model in predicting loan defaulting.
- Precision: The proportion of true positives among all positive predictions.
- Recall: The proportion of true positives predicted correctly among all actual positives.
- Specificity: The proportion of true negatives among all negative predictions.
- ROC AUC score: The area under the receiver operating characteristic curve, which measures the model's ability to distinguish between default and non-default cases.
-
