#'#'%%writefile app.py
import streamlit as st
import numpy as np
import datetime as dt
import pickle
import gzip

from classes import *


with gzip.open('model.pkl.gz', 'rb') as f:
        model = pickle.load(f)


# Define function to calculate age group
def calculate_age_group(age):
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
  df['perc_adult_life_employed'] = (df['yrs_employed'] / (df['age'] - 18))
  ##change days id publish from yrs to days and make negative

  return df
    
  # Calculate balance statistics
  
  # Here put the logic for your prediction
  #prediction = 0
  #return prediction

# Define your Streamlit app
def run():

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
                                              value='')
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
        st.dataframe(df)

        # Display the prediction result
        #st.success(f'The probability of default is: {result}')
        #st.markdown("## Here are your selected features:")
        #st.markdown(feature_dict)
    else:
        st.markdown('## Enter the feature values and click "Predict" to get the default probability.')
    
if __name__=='__main__':
    run()
