import numpy as np
import pandas as pd
import datetime as dt
import pickle
import gzip#
import matplotlib.pyplot as plt
import statsmodels.api as sm

import streamlit as st

#import classes
from helpers.classes import Dataset, Model


@st.cache_resource(ttl=None) #allow model caching
def load_pickles():
    with gzip.open('model_pickle_files/model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    with open('model_pickle_files/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_pickle_files/label_encoders.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    #return loaded objects
    return model, scaler, label_encoder


def safe_divide(numerator, denominator):
    """
    Define function on pandas series that may conatin zeros 
    """
    x =  np.where(denominator != 0, numerator / denominator, 0)
    return x

# Define function to calculate age group
def calculate_age_group(age):
    """
    Discretise age into categories
    """
    if age < 30:
        return 'under 30'
    elif age < 45:
        return '30-45'
    elif age < 60:
        return '45-60'
    else:
        return '60+'

# Define function to calculate debt-to-credit ratio

# Function to calculate statistics on balance and weighted balance
def calculate_balance_statistics(feature_dict):
    balance_string = feature_dict['balance_string']

    if not balance_string:
        feature_dict['average_balance'] = np.nan
        feature_dict['average_weighted_balance'] = np.nan
        feature_dict['maximum_weighted_balance'] = np.nan
    else:
        balance_values = [value.strip() for value in balance_string.split(',')]

        balances = []
        for value in balance_values:
            try:
                balance = float(value)
                balances.append(balance)
            except ValueError:
                balances.append(np.nan)

        feature_dict['average_balance'] = np.nanmean(balances)

        weighted_balances = [balance / (i + 1) for i, balance in enumerate(balances) if not np.isnan(balance)]
        feature_dict['average_weighted_balance'] = np.nanmean(weighted_balances)
        feature_dict['maximum_weighted_balance'] = np.nanmax(weighted_balances)

    return feature_dict
# Define function to perform prediction

def feature_dict_to_df(feature_dict):
    feature_dict = calculate_balance_statistics(feature_dict)
    df = pd.DataFrame(feature_dict, index=[0])
    #use dictionarys to change categorical values to how they exist in training df
    df['gender']=df['gender'].replace({'Male': 'M', 'Female': 'F'})
    df['name_education_type'] =df['name_education_type'].replace({'College/University degree':'Higher education',
                                                                    'Secondary/High School':'Secondary / secondary special'})
    df['name_income_type']=df['name_income_type'].replace({'Salary':'Working','Divedends':'Businessman'})
    df['reg_city_not_work_city'] = df['reg_city_not_work_city'].replace({'Yes':1, 'No':0})
    df['age_group'] = df['age'].apply(calculate_age_group)
    df['days_id_publish'] = df['days_id_publish'].apply(lambda x: -365.25 * x)
    df['days_last_phone_change'] = df['yrs_last_phone_change'].apply(lambda x: -365.25 * x)
    df['perc_adult_life_employed'] = safe_divide(df['yrs_employed'], (df['age'] - 18))
    df['credit_income_ratio'] = safe_divide(df['credit'], df['income'])
    df['yrs_wroking'] = (df['age'] - 18)
    df['perc_adult_life_employed'] = safe_divide(df['yrs_employed'], df['yrs_wroking'])
    df['prop_credit_rejected']=safe_divide(df['total_rejected_apps'],df['total_credit_apps'])
    df['debt_to_credit_ratio']=safe_divide(df['credit_debt'],df['credit_sum'])
    
    return df

def predict_default_prob(df, model, scaler, label_encoders):
    data = Dataset(
        df, 
        is_test=True,
        scaler=scaler,
        label_enocder_dict =label_encoders,
        target=None)

    data.preprocess(final_X_cols = model.feature_names)
    y_pred = model.predict_prob(new_dataset=data)

    return y_pred


    
  # Calculate balance statistics
  
  # Here put the logic for your prediction
  #prediction = 0
  #return prediction

def fit_binomial_reg(model):
    """Fit a binomial regression model and return it."""
    x = model.y_pred
    y = model.y_test
    binomial_reg = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
    return binomial_reg

def plot_glm_prob(binomial_reg, model, predicted_prob,  figsize=(10, 4.5)):        
    """Generate the probability plot based on the binomial regression model."""


    # Generate predicted values for plotting the line
    x_pred = np.linspace(0, 1, 500)
    y_pred = binomial_reg.predict(sm.add_constant(x_pred))
    print(x_pred.shape)

    # Find the regression line value at the predicted probability
    #predicted_prob_reshaped = np.reshape(predicted_prob, (1, ))
    #print(predicted_prob_reshaped.shape)
    # Find the regression line values at the predicted probabilities
    pred_actual_values = binomial_reg.predict(sm.add_constant(pd.Series([1, predicted_prob])))
    print(pred_actual_values)
    reg_line_value = pred_actual_values[1]
    print(f"binomial_reg predicted value is {reg_line_value}")
    y_mean= model.y_test.mean()
    
    # Find the x value (x_intersection) where y_pred is closest to y_mean
    idx = (np.abs(y_pred - y_mean)).argmin()
    x_intersection = x_pred[idx]

    # Plot the data points and the regression line
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_pred, y_pred, color='r', label='Regression Line',linewidth=1.5)
    ax.plot([0, x_intersection], [y_mean, y_mean], color='k', linestyle='--', label='Overall Defaulting Rate',linewidth=0.9)
    ax.plot([x_intersection, x_intersection], [0, y_mean], color='k', linestyle='--',linewidth=0.9)
    ax.plot([predicted_prob, predicted_prob], [0, reg_line_value], color='b', linestyle='--', label='Expected Default Probability',linewidth=1.3)
    ax.plot([0, predicted_prob], [reg_line_value, reg_line_value], color='b', linestyle='--',linewidth=1.3)

    ##set ylim if specified
    ax.set_xlim([0, max(x_pred)])
    ax.set_ylim([0, max(y_pred)])

    if reg_line_value >= y_mean:
      fc = reg_line_value/y_mean
      string = f"({fc:.1f}x higher risk than average)"
    else:
      fc = y_mean/reg_line_value
      string =f"({fc:.1f}x lower risk than average)"


    # Add plot labels and legend
    ax.set_title(f"Model Predicted Default Probability: {predicted_prob*100:.1f}%\nExpected Default Probability: {reg_line_value*100:.1f}% {string}")
    ax.set_xlabel('XGBoost Model Predicted Default Probability')
    ax.set_ylabel('Actual Credit Default Probability')
    ax.legend()

    return fig, reg_line_value

# Define your Streamlit app
def run():
    #load pickle files
    model, scaler, label_encoder = load_pickles()

    # Using markdown for formatted text
    st.markdown("# Loan Default Prediction App")
    
    # Create a sidebar for user input
    st.sidebar.markdown('## User Input')
    
    # Get user inputs from Streamlit sliders and dropdowns
    feature_dict = {}

    feature_dict['gender'] = st.sidebar.selectbox('Which gender do you identify as?', 
                                                  ['Male','Female','Non-binary','Prefer Not to Say'])      
    feature_dict['age'] = st.sidebar.slider('How old are you?', 
                                            min_value=18, max_value=100, value=30, step=1)
    ##create a variable to start dtae pickers
    current_date = dt.date.today()
    date_start = current_date - dt.timedelta(days=(feature_dict['age']+1)*365.25)
    feature_dict['name_education_type'] = st.sidebar.selectbox('What is your highest level of formal education?', 
                                                              ['College/University degree', 
                                                               'Secondary/High School',
                                                               'Lower Secondary'])
    job_start_date = st.sidebar.date_input('When did you start working for your current employer?',
                                           min_value = date_start, max_value=current_date, value = current_date)
    feature_dict['yrs_employed'] = (current_date - job_start_date).days / 365.25
    feature_dict['name_income_type'] = st.sidebar.selectbox('Which of the following best describes your main source of income', 
                                                            ['Student','Pensioner','Salary','Divedends','other'])

    feature_dict['yrs_last_phone_change'] = st.sidebar.slider('How many years since your phone number was last changed?', 
                                                    min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    last_id_change_date = st.sidebar.date_input('When did you last change the ID document that you will use for the loan application?',
                                                 min_value = date_start, max_value=current_date)
    feature_dict['days_id_publish'] = (last_id_change_date-dt.date.today()).days
    feature_dict['reg_city_not_work_city'] = st.sidebar.selectbox('Is your registered address in the same city as your work address',
                                                                  ['Yes', 'No'])
      
    ###credit - related questions####
    feature_dict['income'] = st.sidebar.slider('What is your annual income (US$)?', 
                                                min_value = int(1e4), max_value = int(1e6), value = int(5e4), 
                                               step=5000, format="%d")
    feature_dict['credit'] = st.sidebar.slider('How much would you like to borrow (US$)?',
                                               min_value = 1000, max_value = int(5e6), value = int(feature_dict['income']*3), 
                                               step=1000, format="%d")
    feature_dict['balance_string'] = st.sidebar.text_input("Input monthly credit card balances separated by a comma (e.g. '5000, 6000, 5500, 7000, 8000'), starting with the most recent (leave blank if not applicable)",
                                              value='Please enter values in US$ - eg. 3500,4500,3000,2500,5000,etc..')
    feature_dict['average_days_since_credit_update'] = st.sidebar.slider('Average Days Since Credit Update (across all forms of credit)', 
                                                                         min_value=0, max_value=365, value=30, 
                                                                         step=1, format="%d" )

    feature_dict['credit_sum'] = st.sidebar.slider('Enter the total amount of credit available to you across all forms, including credit cards, car loans, and mortgages.',
                                                   min_value=0, max_value=int(1e6), value=int(1e4), step=1000)
    feature_dict['credit_debt'] = st.sidebar.slider('Enter the total amount of debt you currently owe, considering all outstanding balances on credit cards, car loans, and mortgages.', 
                                                    min_value=0,
                                                    max_value=feature_dict['credit_sum'], 
                                                    value=0, step = 500, format="%d")
    
    feature_dict['total_credit_apps'] = st.sidebar.slider('In total how many previous credit applications have you made?',
                                                          min_value = 0, max_value = 100, value=10, step=1, format="%d")
    feature_dict['total_rejected_apps'] = st.sidebar.slider('How many of these previous credit applications were rejected?',
                                                          min_value = 0, max_value = feature_dict['total_credit_apps'], 
                                                          value=0, step=1, format="%d")
    feature_dict['perc_late'] = st.sidebar.slider('What percentage of your previous credit installments have been paid late',
                                                  min_value = 0, max_value = 100, 
                                                  value=0, step=1)
    feature_dict['perc_underpaid'] = st.sidebar.slider('What percentage of your previous credit installments have been under-paid',
                                                       min_value = 0, max_value = 100, 
                                                       value=0, step=1, format="%d")

    use_receivables = st.sidebar.checkbox('Have you had recievables on previous credit? Usually only relavent for business loans')                                                    
    if use_receivables:
      feature_dict['avg_receivable_sum'] = st.sidebar.slider('Average receivable sum across all previous credits', 
                                                      min_value = 0.0, max_value = int(1e6), step=1000)
      feature_dict['avg_weighted_receivable_sum'] = st.sidebar.slider('Average weighted receivable sum across all previous credits. This is calculated as the average of the recievable amount divided by the how many months ago it was due. This effectively weights the receivable by how long ago it was due, giving recent receivables more weight', 
                                                      min_value = 0.0, max_value = int(1e6), step=1000)
    else:
      feature_dict['avg_receivable_sum'] =0
      feature_dict['avg_weighted_receivable_sum'] =0



    
    
    result = ""
    
    # When 'Predict' is clicked, make the prediction and store it
    if st.sidebar.button("Predict"):
        df = feature_dict_to_df(feature_dict)
        # st.dataframe(df)
        result = predict_default_prob(df, model, scaler, label_encoder)
        result = result[0]

        # Display the prediction result
        st.success(f'Your model estimated probability of default is: {result*100:.1f}%')

        glm = fit_binomial_reg(model)
        fig, pred_rate = plot_glm_prob(glm, model, result)
        st.divider()
        st.markdown('## Model predicted probability vs Real-world default rate')
        st.markdown((
            "Based on your inputs, our **machine learning model has estimated your defaulting probability as "
            f"{result*100:.1f}%** chance of defaulting. But how does this relate to real-world results? "
            "\n\nWe have used a test set (data that our model did not have access to during training) of 46,000 "
            "credit applications, including 3,700 which defaulted, and evaluated the ability of our model to "
            "accurately predict defaulting. Even if our model is returns a 99.9% probability of loan defaulting, "
            "we would still only expect that applicant to have a ~40\% default rate. However this is 5x the average" 
            " defaulting rate in our test set. \n\n **[More info on model training and evaluation](https://github.com/hpink97/loan_default_predictor/blob/main/model_training/03_ml_preprocessing_and_training.ipynb)**"
        ))
        st.pyplot(fig)
        st.markdown(
            f'### Based on this analysis, we expect that a model predicted default probability of {result*100:.1f}% '
            f'equates to an actual default probability of {pred_rate*100:.1f}%'
        )

    else:
        st.markdown('## Enter the feature values and click "Predict" to get the default probability.')

if __name__=='__main__':
    run()
