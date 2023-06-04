# Loan Default Predictor

This repository, `loan_default_predictor`, is designed to predict whether or not a loan will default. The project uses a Kaggle dataset which consists of relational tables and goes through extensive exploratory data analysis (EDA), feature aggregation, feature engineering, and machine learning modeling.

## Dataset

The data used in this project is available on Kaggle at [Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data). The dataset contains seven relational tables:

1. **application_{train|test}.csv**: This is the main table, split into training (with TARGET) and testing (without TARGET) sets. Each row represents one loan in the data sample.

2. **bureau.csv**: Contains the client's previous credits provided by other financial institutions that were reported to the Credit Bureau. For every loan in our sample, there are as many rows as the number of credits the client had in the Credit Bureau before the application date.

3. **bureau_balance.csv**: Contains monthly balances of previous credits in the Credit Bureau. This table has one row for each month of history of every previous credit reported to the Credit Bureau.

4. **POS_CASH_balance.csv**: Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit. This table has one row for each month of history of every previous credit in Home Credit related to loans in our sample.

5. **credit_card_balance.csv**: Monthly balance snapshots of previous credit cards that the applicant has with Home Credit. This table has one row for each month of history of every previous credit in Home Credit related to loans in our sample.

6. **previous_application.csv**: Contains all previous applications for Home Credit loans of clients who have loans in our sample. There is one row for each previous application related to loans in our data sample.

7. **installments_payments.csv**: Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample. There is one row for every payment that was made, plus one row for each missed payment.

## Structure
The project is structured in the following way:

1. **model_training/01_credit_risk_aggregate_and_merge.Rmd**: This file is dedicated to the initial data pre-processing, including EDA, feature aggregation, and feature engineering.

2. **model_training/02_credit_risk_eda_feature_selection.Rmd**: This file is dedicated to the analysis of each feature's relation to the target variable, which is whether the loan will default or not. 

3. **helpers/classes.py**: This file contains Python classes that have been designed to streamline the machine learning pipeline. It includes classes for data pre-processing, feature selection (`SelectKBest`), model training, hyperparameter tuning (`BayesianOptimisation`), model evaluation (via `ROC-AUC`, `F1`, etc.), and making predictions on new data.

4. **model_training/03_ml_preprocessing_and_training.ipynb**: This file is where the model training takes place. It utilizes the classes defined in `helpers/classes.py` to train a comprehensive model using 300 features, as well as a simplified model using the top 25 features.

5. **app.py**: This is a Streamlit application that loads the pickle files of the simplified model and scaler to make predictions.

## Custom classes

### `Dataset` Class

Our `Dataset` class is a custom utility that streamlines data preprocessing, transformation, and splitting for model training, evaluation, and testing. The class is initialized with a pandas DataFrame and includes methods for imputation, scaling, encoding, and splitting the data.

**Key Functionality**

The `Dataset` class includes the following methods:

- `__init__(self, df, target, is_test=False, label_enocder_dict = None, scaler=None, trained_cols = None)`: Initializes the `Dataset` object. The input dataframe is divided into features (`X`) and targets (`y`). If `is_test=True`, the `y` attribute is set to `None`.

- `preprocess(self, impute_dict=None, final_X_cols= None, imputation_kernel_iterations = 4, imputation_kernel_ntrees = 50)`: Preprocesses the data. It performs basic imputations based on the provided `impute_dict`, decision-tree based imputations on the remaining missing data using the MissForest imputation method, scales numerical data using StandardScaler, performs label encoding for binary variables, one-hot encoding for categorical variables, and finally prunes the dataset to only include columns specified in `final_X_cols`.

- `split_data(self, test_size=0.15,eval_size = 0.15, random_state=42)`: Splits the data into training, evaluation, and testing subsets. The split proportions can be adjusted with the `test_size` and `eval_size` parameters. This method can only be run after the `preprocess` method and not on a test set (`is_test=True`).

**Usage**

An example of the Dataset class usage could look like the following:

```python

df = pd.read_csv('application_train.csv')
dataset = Dataset(df, target='TARGET', is_test=False)

# Define the imputation dictionary
impute_dict = {
    'income': 'mean',...
}
# Preprocess the data
dataset.preprocess(impute_dict=impute_dict)
# Split the data
dataset.split_data(test_size=0.2, eval_size=0.1)
```

This will preprocess the dataset (including filling missing values, encoding categorical variables, and scaling numerical variables) and split it into training, evaluation, and testing subsets. This dataset is now ready to be used for model training.

### `Model` class

The `Model` class is designed to train, optimize, and evaluate binary classifiers using the XGBoost algorithm. This class provides a high-level interface for training XGBoost models, selecting features, hyperparameter optimization, evaluating model performance, and making predictions.

**Class Initialization**

To initialize the `Model` class, you need to provide an instance of the `Dataset` class as an argument. The `Dataset` class should have the following attributes: `X_train`, `X_eval`, `X_test`, `y_train`, `y_eval`, and `y_test`. 

Example:
```python
model = Model(dataset)
```

**Feature Selection**

The `Model` class provides a method `select_features(num_features)` to select the best features from the dataset using the `SelectKBest` method from scikit-learn. The `num_features` parameter specifies the number of features to select. This method internally uses the `f_classif` score function to evaluate the features.

Example:
```python
model.select_features(num_features = 50)
```

**Training the Model**

The `train_model` method in the `Model` class trains the XGBoost classifier model. It incorporates evaluation datasets (`X_eval` and `y_eval`) and early stopping to prevent overfitting. Evaluation datasets are used to assess the model's performance on unseen data during training. They help identify overfitting, where the model performs well on the training data but poorly on new data. Early stopping stops the training process if the model's performance on the evaluation dataset doesn't improve for a specified number of rounds (`early_stopping_rounds`). This prevents excessive tuning to the training data and improves generalization.:

- `xgboost_params`: A dictionary of XGBoost parameters.
- `print_training_evaluation` (optional): Whether to print the training evaluation. Defaults to False.
- `num_boost_round` (optional): The number of boosting rounds or trees to build. Defaults to 700.
- `early_stopping_rounds` (optional): Activates early stopping. Validation metric needs to improve at least once in every `early_stopping_rounds` round(s) to continue training. Defaults to 20.

Example:
```python
xgboost_params = {'max_depth': 5, 'learning_rate': 0.1, ...}
model.train_model(xgboost_params,num_boost_round=700,early_stopping_rounds=20  )
```

**Hyperparameter Optimization**

The `Model` class provides a method `bayesian_hyperparam_optimisation(pbounds, ...)` to perform Bayesian hyperparameter optimization using the `BayesianOptimization` library. The method allows num_features to be defined within `pbounds`, which perform `.select_features()` during the optimisation process. It takes the following parameters:

- `pbounds`: A dictionary containing hyperparameter bounds for the optimization.
- `start_hyperparam` (optional): A dictionary containing initial hyperparameters. Defaults to None.
- `initial_random_search_iterations` (optional): Number of initial iterations to perform random search. Defaults to 8.
- `bayesian_search_iterations` (optional): Number of iterations to perform Bayesian optimization. Defaults to 20.
- `retrain_with_best_params` (optional): Whether to retrain the model with the best parameters found. Defaults to True.

Example:
```python
pbounds = {'max_depth': (3, 10), 'learning_rate': (0.01, 0.1), 'num_features':(10,100)}
model.bayesian_hyperparam_optimisation(pbounds)
```

**Model Evaluation**

The `Model` class provides several methods to evaluate the performance of the trained model:

- `roc_auc()`: Calculate the ROC AUC score for the model.
- `evaluate_model(opt_thresh_search_precision)`: Evaluate the model performance using various metrics. This method also searches for the optimal threshold for binary target classification where the F1 score is highest.
- `plot_roc_auc()`: Plot the receiver operating characteristic (ROC) curve for the model.
- `plot_feature_importance(n_features)`: Plot the feature importance for the model. The `n_features` parameter specifies

## Models

Two models were trained in this project:

1. **Full Model**: This comprehensive model uses 300 features and achieved a ROC-AUC score of 0.775. It utilizes features that are difficult to collect additional data for, such as those from unknown external sources. 

```python
model_full.evaluate_model()
```
Result:
```
Optimal Threshold: 0.660
F1 Score: 0.327
Accuracy: 0.849
Precision: 0.255
Recall (Sensitivity): 0.457
Specificity (True Negative Rate): 0.883
ROC AUC Score: 0.777
Balanced Accuracy: 0.670
```

2. **Simplified Model**: This model is a pared-down version of the full model, utilizing only the best 25 features. These are features for which it is feasible to collect new data. The simplified model is used in the Streamlit application, as it is more scalable and applicable in a practical setting.

```python
model_simple.evaluate_model()
```
Result:
```
Optimal Threshold: 0.690
F1 Score: 0.255
Accuracy: 0.807
Precision: 0.185
Recall (Sensitivity): 0.411
Specificity (True Negative Rate): 0.842
ROC AUC Score: 0.702
Balanced Accuracy: 0.626
```

## Streamlit Application

The Streamlit application (`app.py`) is a user-friendly tool that can load the simplified model and its associated scaler (stored as pickle files) and make predictions. This allows for practical, real-time use of the model and makes it accessible for non-technical stakeholders.
