import pandas as pd
import numpy as np 
##import gc #free up memory
import matplotlib.pyplot as plt
import miceforest as mf ##forest based imputation
import statsmodels.api as sm


import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score,roc_curve

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold



class Dataset:
    """
    ## `Dataset` class for ML preprocessing

    The `Dataset` class is designed to handle the preprocessing and splitting of a dataset for machine learning tasks. Here is a summary of its functionality:

    1. Initialization: The class takes in a pandas DataFrame (`df`) representing the dataset and the target variable (`target`). It also accepts additional parameters like `is_test` to indicate if the dataset is a test set, `scaler` for scaling numeric columns, and `trained_cols` for indicating specific columns to be used in the final dataset.

    2. Preprocessing: The `preprocess()` method performs preprocessing tasks on the dataset. It includes basic imputations for missing values, smart imputations using the `miceforest` package for remaining missing values, scaling of numeric columns, and label encoding for binary columns. It also performs one-hot encoding for categorical columns.

    3. Splitting Data: The `split_data()` method splits the preprocessed dataset into training, evaluation, and testing sets. It takes parameters like `test_size` and `eval_size` to control the size of the test and evaluation sets, respectively. It prints information about the sizes and positive rates of each split.
    """
    def __init__(self, df, target,is_test=False,
                 label_enocder_dict = None, scaler=None):
      input_df = df.copy()
      self.is_test = is_test
      self.target = target
      self.X_train = None
      self.X_test = None
      self.X_eval = None
      self.y_eval = None
      self.y_train = None
      self.y_test = None
      self.preprocessed = False
      self.scaler = scaler
      self.label_encoders = label_enocder_dict
      if self.is_test:
        self.y = None
        self.X = input_df
      else:
        self.y = input_df[self.target]
        self.X = input_df.drop(columns=[self.target])

    ##method to pre-process the df
    def preprocess(self, impute_dict=None, final_X_cols= None,
                   imputation_kernel_iterations = 4, imputation_kernel_ntrees = 50):
       # Basic imputations
      if impute_dict is not None:
        print(f'Performing basic imputations based on {len(impute_dict)} features supplied impute_dict')
        for col, strategy in impute_dict.items():
          if col not in self.X.columns:
            print(f"Skipping imputation for column '{col}' as it does not exist in the dataset.")
            continue
          if strategy == 'mean':
              self.X[col].fillna(self.X[col].mean(), inplace=True)
          elif strategy == 'median':
              self.X[col].fillna(self.X[col].median(), inplace=True)
          elif isinstance(strategy, str) and strategy.startswith('percentile_'):
              percentile = float(strategy.split('_')[1])
              self.X[col].fillna(self.X[col].quantile(percentile / 100), inplace=True)
          elif isinstance(strategy, (int, float)):
              self.X[col].fillna(strategy, inplace=True)
          else:
              raise ValueError(f"Invalid imputation strategy for column '{col}'.")
    #find numeric columns
      numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
      #print(numeric_cols)

      ## smart impuations - decision tree based method
      count_NA = self.X.isna().sum()
      remaining_NA_cols = count_NA[count_NA>0].shape[0]
      #print(remaining_NA_cols)
      if  remaining_NA_cols> 0:
        print(f'Performing decision-tree based imputations of {remaining_NA_cols} remaining features with missing data')
        kernal = mf.ImputationKernel(
            self.X[numeric_cols],
            random_state=42
            )
        # Run the MICE algorithm for 2 iterations
        kernal.mice(iterations=imputation_kernel_iterations,
                    n_estimators=imputation_kernel_ntrees)
        X_numeric_imputed = kernal.complete_data()
        self.X[numeric_cols] = X_numeric_imputed
      
      ##scale numeric cols
      print('Scaling numeric data')
      if self.scaler is None:
        self.scaler = StandardScaler()
        self.X[numeric_cols] = self.scaler.fit_transform(self.X[numeric_cols])
      else:
        scaler_trained_features = self.scaler.feature_names_in_
        missing_cols = set(scaler_trained_features) - set(self.X.columns)
        if len(missing_cols) >0:
          print(f"{len(missing_cols)} variables that scaler was originally trained on are missing - will replace with zeros")
          for col in missing_cols:
            self.X[col] = 0
        self.X[scaler_trained_features] = self.scaler.transform(self.X[scaler_trained_features])
  
      
      # Perform label encoding for binary columns
      print('One-hot-encoding categorical vars')
      if self.label_encoders is None:
        binary_cols = [col for col in self.X.columns if self.X[col].nunique() == 2]
        self.label_encoders = {}
        # Label encode binary columns
        for col in binary_cols:
          label_encoder = LabelEncoder()
          self.X[col] = label_encoder.fit_transform(self.X[col])
          # Store the label encoder for later use
          self.label_encoders[col] = label_encoder
      ##
      else:
        binary_cols = self.label_encoders.keys()
        for col in binary_cols:
          self.X[col] = self.label_encoders[col].transform(self.X[col])

        
      
      # Perform one-hot encoding for categorical columns
      categorical_cols = [col for col in self.X.columns if self.X[col].dtype == 'object' and col not in binary_cols]
      self.X = pd.get_dummies(self.X, columns=categorical_cols)

      if final_X_cols is not None:
        print('splicing dataset to include only final_X_cols columns')
        missing_cols = set(final_X_cols) - set(self.X.columns)
        for col in missing_cols:
          self.X[col] = 0
        self.X = self.X[final_X_cols]

      self.preprocessed = True
    

    def split_data(self, test_size=0.15,eval_size = 0.15, random_state=42):
        if not self.preprocessed:
          raise RuntimeError("Data has not been preprocessed. Please run the preprocess method.")
        
        if self.is_test:
          raise RuntimeError("Cannot run split_data() method on a test set")

        
        X_train_eval, self.X_test, y_train_eval, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        eval_split_size = eval_size/(1-test_size)
        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            X_train_eval, y_train_eval, test_size=eval_split_size, random_state=random_state)

        print(f"{self.X_train.shape[0]} training samples, {self.X_eval.shape[0]} evaluation samples and {self.X_test.shape[0]} testing samples")
        print(f"{self.y_train.sum()} ({self.y_train.mean()*100:.3f}%) positives in training set")
        print(f"{self.y_eval.sum()} ({self.y_eval.mean()*100:.3f}%) positives in evaluation set")
        print(f"{self.y_test.sum()} ({self.y_test.mean()*100:.3f})% positives in testing set")



class Model:
    ##initialise the model class - taking instance of `Dataset` class an input
    def __init__(self, dataset_class):

      """
      Initialize the model.
      Args:
        dataset_class (Dataset): An instance of the Dataset class. It should have attributes X_train, X_test, y_train, y_test.
      """
      if not isinstance(dataset_class.X_train, pd.DataFrame) or not isinstance(dataset_class.X_test, pd.DataFrame):
          raise TypeError("X_train and X_test must be pandas DataFrames")
      if not isinstance(dataset_class.y_train, pd.Series) or not isinstance(dataset_class.y_test, pd.Series):
          raise TypeError("y_train and y_test must be pandas Series")
      ## take train-eval-test split datasets from input
      self.X_train = dataset_class.X_train
      self.X_eval = dataset_class.X_eval
      self.X_test = dataset_class.X_test
      self.y_train = dataset_class.y_train
      self.y_eval = dataset_class.y_eval
      self.y_test = dataset_class.y_test
      ##set up attributes to be used later
      self.y_pred = None
      self.xgboost_params = None
      self.feature_names = dataset_class.X_train.columns
      self.model = None
      self.is_model_trained = False


    ##define method to select best n features in dataset
    ##basically a wrapper for SelectKBest from sklearn

    def select_features(self, num_features):

      """
      Select the best features from the dataset using SelectKBest from sklearn.
      Args:
        num_features (int or float): The number of features to select.
      """
      if not isinstance(num_features, (int, float)):
          raise TypeError("num_features must be an int or float")

      # Create VarianceThreshold object
      constant_filter = VarianceThreshold(threshold=0)

      # Fit VarianceThreshold object to data, then get the support mask
      constant_filter.fit(self.X_train)
      constant_support = constant_filter.get_support()

      # Get the columns with >0 variance
      non_constant_columns = self.X_train.columns[constant_support]

      # Round num_features to the nearest integer
      num_features = max(min(round(num_features), len(non_constant_columns)), 2)

      # Perform feature selection on columns with >0 variance
      selector = SelectKBest(score_func=f_classif, k=num_features)
      selector.fit(self.X_train[non_constant_columns], self.y_train)
      mask = selector.get_support()

      # Get selected feature indices
      selected_indices = np.where(mask)[0]

      # Map indices back to column names
      self.feature_names = non_constant_columns[selected_indices]

      #print(f'Completed feature selection for best {num_features}')


    ## defne method to train the classifier
    def train_model(self, 
                    xgboost_params,
                    print_training_evaluation=False ,
                    num_boost_round=700,
                    early_stopping_rounds=20):
      """Train the XGBoost classifier model.
      Args:
          xgboost_params (dict): A dictionary of XGBoost parameters.
          print_training_evaluation (bool, optional): Whether to print the training evaluation. Defaults to False.
          num_boost_round (int, optional): The number of boosting rounds or trees to build. Defaults to 700.
          early_stopping_rounds (int, optional): Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training. Defaults to 20.
      """
      self.xgboost_params = xgboost_params

      ##create xgb Matrix objects for datasets
      #print('subsetting datasets of selected')
      x_train = self.X_train.loc[:, self.feature_names]
      x_eval = self.X_eval.loc[:, self.feature_names]

      


      #print('making xgb.DMatrix objects from datasets')
      dtrain = xgb.DMatrix(x_train, label=self.y_train)
      ##
      deval = xgb.DMatrix(x_eval, label = self.y_eval)
      self.model = xgb.train(self.xgboost_params,
                             dtrain = dtrain, 
                             evals=[(deval, 'eval')],
                             verbose_eval=print_training_evaluation,
                             num_boost_round=num_boost_round,
                             early_stopping_rounds=early_stopping_rounds )
      print(f'Model training completed on {self.X_train.loc[:, self.feature_names].shape[1]} features. Best evaluation score (ROC-AUC:{self.model.best_score:.3f}) was obtained at iteration {self.model.best_iteration}')
      self.is_model_trained = True

    def _objective_function(self,
                            learning_rate,
                            max_depth, 
                            scale_pos_weight,
                            subsample,
                            colsample_bytree,
                            colsample_bynode,
                            min_child_weight, 
                            num_features):
      """
      Defines the objective function for the XGBoost model.
      This will be utilised in a bayesian search for optimal hyperparams
      See `bayesian_hyperparam_optimisation()` method.

      Args:
          learning_rate (float): Learning rate for the XGBoost model.
          max_depth (int): Maximum depth of a tree for the XGBoost model.
          scale_pos_weight (float): Controls the balance of positive and negative weights.
          subsample (float): Subsample ratio of the training instances.
          colsample_bytree (float): Subsample ratio of columns when constructing each tree.
          colsample_bynode (float): Subsample ratio of columns for each node.
          min_child_weight (int): Minimum sum of instance weight (hessian) needed in a child.
          num_features (int): The number of features in the dataset.
      """
      
      if self.is_model_trained:
          raise RuntimeError("Model cannot be trained before hyperparam_tuned")
      xgb_params = {
          'max_depth': int(max_depth),
          'learning_rate': learning_rate,
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'scale_pos_weight': scale_pos_weight,
          'subsample':subsample,
          'colsample_bynode':colsample_bynode,
          'min_child_weight': min_child_weight,
          'colsample_bytree':colsample_bytree,
          'eval_metric':'auc'
          }
      #select features if number inputted lower than number of features
      if num_features < self.X_train.shape[1]:
        self.select_features(num_features)
      self.train_model(xgb_params,
                       num_boost_round=250,
                       early_stopping_rounds=5,
                       print_training_evaluation=False)
      
      self.is_model_trained = False
          
          # Return ROC-AUC value for BayesianOptimization to maximise
      return self.model.best_score

    def bayesian_hyperparam_optimisation(self, pbounds, 
                                         start_hyperparam = None,
                                         initial_random_search_iterations = 8,
                                         bayesian_search_iterations = 20,
                                         retrain_with_best_params=True):
      """Perform Bayesian hyperparameter optimization.

      Args:
          pbounds (dict): Dictionary containing hyperparameter bounds for the optimization.
          start_hyperparam (dict, optional): Dictionary containing initial hyperparameters. Defaults to None.
          initial_random_search_iterations (int, optional): Number of initial iterations to perform random search. Defaults to 8.
          bayesian_search_iterations (int, optional): Number of iterations to perform Bayesian optimization. Defaults to 20.
          retrain_with_best_params (bool, optional): Whether to retrain the model with the best parameters found. Defaults to True.
      """

      print('Validating hyperparamater inputs')
      hyper_params = ['max_depth','learning_rate','scale_pos_weight',
                      'subsample','colsample_bynode','min_child_weight',
                      'colsample_bytree','num_features']
      required_params = hyper_params[0:7]
      all_present = all(item in pbounds for item in required_params)

      if not all_present:
        raise RuntimeError("Some required hyperparams for Bayesian Optimisation are missing")
      if 'num_features' not in pbounds:
        ##if num features not set, set it a high number.
        pbounds['num_features'] = (1e6, 1e6+1)

      
      
      optimizer = BayesianOptimization(
          f=self._objective_function,
          ##subset input for allowed hyperparams
          pbounds={param: pbounds[param] for param in hyper_params},
          random_state=42,
          verbose=2
          )
      
      if start_hyperparam is not None:
        all_present_start_point = all(item in start_hyperparam for item in required_params)
        if all_present_start_point:
          print(f'Starting optimisation at specified paramaters: {start_hyperparam}')
          optimizer.probe({param: start_hyperparam[param] for param in pbounds.keys()})
      
      print('Performing hyperparamater optimisation')
      optimizer.maximize(init_points=initial_random_search_iterations, 
                         n_iter=bayesian_search_iterations)
      
      best_params = optimizer.max['params']
      best_params['max_depth'] = int(best_params['max_depth'])
      best_params['objective'] ='binary:logistic'
      best_params['eval_metric']='auc'

      del best_params["num_features"]

      if retrain_with_best_params:
        self.train_model(best_params,
                          num_boost_round=1250,
                          early_stopping_rounds=50,
                          print_training_evaluation=False)
      
      return optimizer

    def roc_auc(self):
      """Calculate ROC AUC score for the model.
      """

      if not self.is_model_trained:
          raise RuntimeError("Model must be trained before it can be evaluated")
      if self.y_pred is None:
        dtest = xgb.DMatrix(self.X_test.loc[:, self.feature_names])
        self.y_pred = self.model.predict(dtest)

      roc_auc = roc_auc_score(self.y_test, self.y_pred)

      return roc_auc
    
    def evaluate_model(self, opt_thresh_search_precision=0.01):
      """
      Evaluate the model performance using various metrics.
      This method also searches for the optimal threshold for binary target classification where F1 score is highest
      Args:
          opt_thresh_search_precision (float, optional): Precision for threshold search. Defaults to 0.01.
      """

      if not self.is_model_trained:
          raise RuntimeError("Model must be trained before it can be evaluated")
      if self.y_pred is None:
        dtest = xgb.DMatrix(self.X_test.loc[:, self.feature_names])
        self.y_pred = self.model.predict(dtest)
      ##select best threshold for determining 
      best_threshold = 0
      best_f1 = 0
      
      # Iterate over different threshold values
      for threshold in np.arange(0.01, 1.0, opt_thresh_search_precision):
          y_pred_binary = (self.y_pred >= threshold).astype(int)
          f1 = f1_score(self.y_test, y_pred_binary)

          if f1 > best_f1:
              best_f1 = f1
              best_threshold = threshold
      
      # Convert predicted probabilities to binary predictions based on the best threshold
      y_pred_binary = (self.y_pred >= best_threshold).astype(int)
      
      # Calculate accuracy metrics
      accuracy = accuracy_score(self.y_test, y_pred_binary)
      precision = precision_score(self.y_test, y_pred_binary)
      recall = recall_score(self.y_test, y_pred_binary)
      specificity = recall_score(self.y_test, y_pred_binary, pos_label=0)
      roc_auc = self.roc_auc()
      balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred_binary)
      
      # Print the metrics
      print(f"Optimal Threshold: {best_threshold:.3f}")
      print(f"F1 Score: {best_f1:.3f}")
      print(f"Accuracy: {accuracy:.3f}")
      print(f"Precision: {precision:.3f}")
      print(f"Recall (Sensitivity): {recall:.3f}")
      print(f"Specificity (True Negative Rate): {specificity:.3f}")
      print(f"ROC AUC Score: {roc_auc:.3f}")
      print(f"Balanced Accuracy: {balanced_accuracy:.3f}")

    
    def plot_roc_auc(self):
      if not self.is_model_trained:
          raise RuntimeError("Model must be trained before it can be evaluated")
      if self.y_pred is None:
        dtest = xgb.DMatrix(self.X_test.loc[:, self.feature_names])
        self.y_pred = self.model.predict(dtest)
      
      fpr, tpr, _ = roc_curve(self.y_test, self.y_pred)
      roc_auc = self.roc_auc()
      plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic')
      plt.legend(loc="lower right")
      plt.show()

    def plot_prediction_prob(self,ylim=None, figsize=(10, 4.5)):
      if self.y_pred is None:
        dtest = xgb.DMatrix(self.X_test.loc[:, self.feature_names])
        """Plot a binomial regression line on a scatter plot of the data.

        Parameters
        ----------
        ylim : tuple, optional
            The limits of the y-axis. Defaults to None.
        figsize : tuple, optional
            The size of the plot. Defaults to (10, 4.5).

        Returns
        -------
        None
        """
      ##how much does are predicted y probability value relate to the actual y probability 
      x=self.y_pred
      y= self.y_test

      # Fit the binomial GLM model
      binomial_reg = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()

      # Generate predicted values for plotting the line
      x_pred = np.linspace(x.min(), x.max(), 100)
      y_pred = binomial_reg.predict(sm.add_constant(x_pred))

      # Plot the data points and the regression line
      fig, ax = plt.subplots(figsize=figsize)
      #ax.scatter(x, y, color='b', alpha=0.5, label='Data')
      ax.plot(x_pred, y_pred, color='r', label='Regression Line')
      ax.axhline(self.y_test.mean(), color='k', linestyle='--', label='Overall Defaulting Rate')

      ##set ylim if specified
      if ylim:
          ax.set_ylim(ylim)
      
      # Add plot labels and legend
      ax.set_title('')
      ax.set_xlabel('XGBoost Model Predicted Default Probability')
      ax.set_ylabel('Actual Credit Default Probability')
      ax.legend()

      plt.show()
    
    def plot_feature_importance(self, n_features =None):
      if not self.is_model_trained:
          raise RuntimeError("Model must be trained before feature importances can be evaluated")

      feature_importances = self.model.get_score(importance_type='weight')
      if n_features is None:
        n_features = len(feature_importances)

      sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

      # Extract feature names and importance scores
      features = [x[0] for x in sorted_feature_importances]
      importances = [x[1] for x in sorted_feature_importances]

      # Create a bar plot of sorted feature importances
      plt.figure(figsize=(8, 9.5))
      plt.barh(features[0:(n_features-1)][::-1], 
               importances[0:(n_features-1)][::-1])
      plt.xlabel('Importance')
      plt.ylabel('Features')
      plt.title(f'Top {n_features} Features')
      plt.tight_layout()
      plt.show()

    def predict_prob(self, new_dataset, threshold = None): 
      """
      Generate predictions from the model from new data. 
      If a threshold is provided, method will provide binary predictions. 
      If no threshold is provided (default), probabilities will be outputted 

      Args:
          new_dataset (pd.DataFrame or Dataset): New data to make predictions on.
          threshold (float, optional): Threshold for converting probabilities to binary predictions. Defaults to None 
      """
      ##check that the model has been trained
      if not self.is_model_trained:
          raise RuntimeError("Model must be trained before new predictions can be made")
      ###check that trained features are supplied in the new dataset
      if isinstance(new_dataset, pd.DataFrame):
        if not all(feature in new_dataset.columns for feature in self.feature_names):
            raise ValueError("All feature names must be in the new dataset")
        data = new_dataset.loc[:, self.feature_names]
      elif isinstance(new_dataset, Dataset): 
        if not all(feature in new_dataset.X.columns for feature in self.feature_names):
            raise ValueError("All feature names must be in the new dataset")
        data = new_dataset.X.loc[:, self.feature_names]
      
      ##perform predictions
      dtest = xgb.DMatrix(data)
      y = self.model.predict(dtest)
      if threshold is None:
        return y
      elif isinstance(threshold, float):
        ##ensure threshold is between 0 and 1
        threshold = max(min(float(threshold), 1), 0)
        print(f'obtaining model class using supplied threshold:{threshold}')
        ##convert to class
        y_class = (y >= threshold).astype(int)
        return y_class
      else:
        print('threshold must be supplied as a float between 0 and 1 - returning probabilities instead!')
        return y
      
      

