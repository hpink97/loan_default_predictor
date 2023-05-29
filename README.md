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

## Model Evaluation

The performance of the loan default prediction model is evaluated using various evaluation metrics to assess its accuracy and effectiveness. The [`Model`](03_ml_preprocessing_and_training.ipynb) class provides methods for evaluating the model and visualizing its performance, including:

- F1 score: The harmonic mean of precision and recall, which balances both metrics.
- Accuracy: The overall accuracy of the model in predicting loan defaulting.
- Precision: The proportion of true positives among all positive predictions.
- Recall: The proportion of true positives predicted correctly among all actual positives.
- Specificity: The proportion of true negatives among all negative predictions.
- ROC AUC score: The area under the receiver operating characteristic curve, which measures the model's ability to distinguish between default and non-default cases.

## Results

