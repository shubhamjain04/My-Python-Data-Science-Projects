PYTHON CREDIT CARD CASE STUDY

# IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

path = 'C:/Users/shubh/Desktop/ALABS PYTHON CASE STUDIES/FINAL/1. Predicting Credit Card Spend & Identifying Key Drivers/'

# SETTING DISPLAY OPTIONS

pd.set_option('display.max_columns', 175)
pd.set_option('display.max_rows', 175)
pd.set_option('display.max_info_columns', 175)

# IMPORTING DATASET

credit_data = pd.read_excel(path + 'Data Set.xlsx')

credit_data.head()

credit_data.info()

credit_data.nunique()

# CORRECTING THE DATATYPES OF FEATURES

# CONVERTING ORDINAL/NOMINAL CATEGORICAL AND BINOMIAL CATEGORICAL VARIABLES TO OBJECT TYPE
credit_data.loc[ :, credit_data.nunique() <= 6] = credit_data.loc[ :, credit_data.nunique() <= 6].astype('object')

credit_data.loc[ :, 'pets_birds'] = credit_data.loc[ :, 'pets_birds'].astype('int64') 

credit_data.info()

# CREATING UDF's TO PERFORM EXPLORATORY  DATA  ANALYSIS AND DATA PREPARATION

# UDF TO CREATE UNIVARIATE ANALYSIS FOR NUMERICAL FEATURES
def continuous_var_summary(x) :
    
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(), x.std(), x.var(), 
                      x.min(), x.quantile(0.01), x.quantile(0.05), x.quantile(0.1), x.quantile(0.25), 
                      x.quantile(0.5), x.quantile(0.75), x.quantile(0.9), x.quantile(0.95), x.quantile(0.99), x.max()],
                      index = ['N', 'NMISS', 'TOTAL SUM', 'MEAN', 'MEDIAN', 'STDEV', 'VAR', 'MIN', 'P1', 'P5', 'P10', 'P25',
                               'P50', 'P75', 'P90', 'P95', 'P99', 'MAX'])


# UDF TO CREATE UNIVARIATE ANALYSIS FOR CATEGORICAL FEATURES
def categorical_var_summary(x) :
    
    mode = x.value_counts().sort_values(ascending = False).iloc[0:1, ].reset_index()
    
    return pd.Series([x.count(), x.isnull().sum(), x.nunique(),
                      mode.iloc[0, 0], mode.iloc[0, 1], round(mode.iloc[0, 1]*100/x.count(), 2) ],
                      index = ['N', 'NMISS', 'NUNIQUE', 'MODE', 'FREQ', 'PERCENT'])


# UDF TO IMPUTE MISSING VALUES FOR BOTH NUMERICAL AND CATEGORICAL VARIABLES
def missing_value_imputation(x) :
    
    # Replacing the missing values for continuous variables with mean value
    if(x.dtypes == 'float64') | (x.dtypes == 'int64') :
        x = x.fillna(x.mean())
    
    # Replacing the missing values for categorical variables with mode value
    elif(x.dtypes == 'object') :
        mode = x.value_counts().sort_values(ascending = False).reset_index().iloc[0, 0]
        x = x.fillna(mode)
        
    return x

# COUNT OF MISSING VALUES IN OUR DATA
credit_data.isnull().sum().sum()

# LOGARITHMIC FEATURES
log_features = ['lnlongten', 'lntollmon', 'lntollten', 'lnequipmon', 'lnequipten', 'lncardmon', 'lncardten', 'lnwiremon', 'lnwireten']

# REPLACING THE NULL VALUES OF LOG FEATURES WITH 0 SINCE LOG OF ZERO IN UNDEFINED
credit_data[log_features] = credit_data[log_features].fillna(0)

# COUNT OF MISSING VALUES IN OUR DATA AFTER REPLACING LOG NULL WITH ZERO
credit_data.isnull().sum().sum()

# SEPERATING NUMERICAL AND CATEGORICAL FEATURES

credit_data_num = credit_data.select_dtypes(['int64', 'float64'])
credit_data_num.head()

credit_data_cat = credit_data.select_dtypes('object')
credit_data_cat.head()

# CREATING UNIVARIATE ANALYSIS SUMMARY FOR NUMERICAL FEATURES
univ_analysis = credit_data_num.apply(continuous_var_summary).T
univ_analysis.to_csv(path + 'univ_analysis1.csv')
univ_analysis

# CREATING UNIVARIATE ANALYSIS FOR CATEGORICAL FEATURES
univ_analysis_cat = credit_data_cat.apply(categorical_var_summary).T
univ_analysis_cat.to_csv(path + 'univ_analysis_cat1.csv')
univ_analysis_cat

# APPLYING OUTLIER TREATMENT ON NUMERICAL FEATURES

credit_data_num = credit_data_num.apply(lambda x : x.clip(lower = x.quantile(0.01), 
                                                          upper = x.quantile(0.99)))
credit_data_num

credit_data_num.isnull().sum().sum()


# APPLYING MISSING VALUE IMPUTATION ON NUMERICAL FEATURES

credit_data_num = credit_data_num.apply(missing_value_imputation)
credit_data_num

# CREATING UNIVARIATE ANALYSIS SUMMARY FOR NUMERICAL FEATURES
univ_analysis = credit_data_num.apply(continuous_var_summary).T
univ_analysis.to_csv(path + 'univ_analysis2.csv')

credit_data_cat.isnull().sum().sum()

# APPLYING MISSING VALUE IMPUTATION ON CATEGORICAL FEATURES

credit_data_cat = credit_data_cat.apply(missing_value_imputation)
credit_data_cat


# CREATING UNIVARIATE ANALYSIS FOR CATEGORICAL FEATURES
univ_analysis_cat = credit_data_cat.apply(categorical_var_summary).T
univ_analysis_cat.to_csv(path + 'univ_analysis_cat2.csv')
univ_analysis_cat


# CREATING DUMMIES FOR NOMINAL CATEGORICAL VARIABLES

nominal_cat_vars = ['region', 'jobcat', 'hometype', 'reason', 'card', 'cardtype', 'cardbenefit', 'card2', 'card2type', 'card2benefit', 'bfast', 'internet']

credit_data_cat = pd.get_dummies(credit_data_cat, columns = nominal_cat_vars, drop_first = True, dtype = 'int64')
credit_data_cat


# CONCATENATING PROCESSED NUMERICAL AND CATEGORICAL FEATURES FOR FURTHER ANALYSIS

credit_data_final = pd.concat([credit_data_cat, credit_data_num], axis = 1)
credit_data_final

credit_data_final.info()


# CREATING TARGET VARIABLE TOTAL_SPEND BY ADDING CARD_SPENT AND CARD2_SPENT

credit_data_final['total_spend'] = credit_data_final['cardspent'] + credit_data_final['card2spent'] 
credit_data_final

# CORRELATION MATRIX

plt.figure(figsize = (15, 10))
sns.heatmap(credit_data_final.corr())
plt.show()

corr = credit_data_final.corr()
corr.to_csv(path + 'corr.csv')
corr

# DISTRIBUTION OF TARGET VARIABLE

# DISTRIBUTION OF ORIGINAL TARGET VARIABLE
credit_data_final['total_spend'].hist()

# DISTRIBUTION OF LOG TRANSFORMED TARGET VARIABLE
np.log(credit_data_final['total_spend']).hist()

# DISTRIBUTION OF SQUARE ROOT TRANSFORMED TARGET VARIABLE
np.sqrt(credit_data_final['total_spend']).hist()

credit_data_final['total_spend'].skew(), np.log(credit_data_final['total_spend']).skew(), np.sqrt(credit_data_final['total_spend']).skew()

# SINCE THE SKEWNESS OF LOG TRANSFORMED TARGET VARIABLE IS MINIMUM, WE WILL USE LOG TRANSFORMED TOTAL_SPEND FOR MODELLING
credit_data_final['ln_total_spend'] = np.log(credit_data_final['total_spend'])
credit_data_final.head()

# REMOVING IRRELEVANT VARIABLES FROM OUR DATA
credit_data_final.drop(columns = ['custid', 'birthmonth', 'commutecat', 'cardspent', 'card2spent'], inplace = True)

credit_data_final.shape

credit_data_final.head()

# APPLYING F-REGRESSION TO IDENTIFY SIGNIFICANT VARIABLES 

from sklearn.feature_selection import f_regression

feature_columns = credit_data_final.columns.difference(['total_spend', 'ln_total_spend'])

features = credit_data_final[feature_columns]
target   = credit_data_final['ln_total_spend']

f_values, p_values = f_regression(features, target)

f_regression_op = pd.DataFrame([feature_columns, f_values.round(2), p_values.round(4)]).T
f_regression_op.columns = ['Features', 'F_values', 'P_value']
feature_columns = list(f_regression_op.loc[f_regression_op['P_value'] < 0.05, 'Features'])

f_regression_op

print(feature_columns)

len(feature_columns)

feature_columns = ['addresscat', 'agecat', 'bfast_2', 'bfast_3', 'callid', 'callwait', 'card2_2', 'card2_3', 'card2_4', 'card2_5', 'card2benefit_3', 'card_2', 'card_3', 'card_4', 'card_5', 'carown', 'carvalue', 'confer', 'creddebt', 'ebill', 'edcat', 'employ', 'forward', 'gender', 'homeown', 'hometype_2', 'hourstv', 'lninc', 'inccat', 'internet_3', 'internet_4', 'jobcat_2', 'jobcat_3', 'jobsat', 'lncardten', 'lncreddebt', 'lnequipten', 'lnlongten', 'lnothdebt', 'longten', 'multline', 'news', 'othdebt', 'owncd', 'owndvd', 'ownfax', 'owngame', 'ownipod', 'ownpc', 'ownpda', 'owntv', 'ownvcr', 'pager', 'polcontrib', 'reason_2', 'reason_4', 'region_5', 'response_03', 'retire', 'spousedcat', 'tollfree', 'tollten', 'voice', 'vote', 'wireless', 'wireten']

# APPLYING VARIANCE INFLATION FACTOR TO REMOVE MULTICOLLINEARITY

from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

model_param = 'ln_total_spend ~' + '+'.join(feature_columns)
y, X = dmatrices(model_param, credit_data_final, return_type = 'dataframe')
vif = pd.DataFrame()

vif['features'] = X.columns
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif.round(1)

feature_columns = list(set(vif.loc[ : , 'features']).difference(['Intercept']))

vif

print(feature_columns)

# SPLITTING THE DATA FOR TRAINING AND TESTING

from sklearn.model_selection import train_test_split

credit_data_train, credit_data_test = train_test_split(credit_data_final, test_size = 0.3, random_state = 12345)

credit_data_train.shape, credit_data_test.shape

# FEATURES TO BE USED FOR MODEL BUILDING
model_param = ['carown', 'jobcat_2', 'news', 'creddebt', 'lninc', 'spousedcat', 'gender', 'lnequipten', 'addresscat', 'lncreddebt', 'wireless', 'edcat', 'card2_4', 'lnothdebt', 'card_5', 'card_3', 'carvalue', 'response_03', 'bfast_2', 'ownpc', 'othdebt', 'internet_4', 'confer', 'card2_5', 'card_2', 'hourstv', 'longten', 'hometype_2', 'lnlongten', 'jobcat_3', 'ownipod', 'owngame', 'callwait', 'retire', 'internet_3', 'ownpda', 'reason_4', 'ownfax', 'bfast_3', 'lncardten', 'employ', 'vote', 'card2benefit_3', 'card_4', 'owndvd', 'agecat', 'jobsat', 'wireten', 'polcontrib', 'region_5', 'multline', 'pager', 'reason_2', 'owntv', 'owncd', 'voice', 'homeown', 'callid', 'inccat', 'card2_3', 'ownvcr', 'tollfree', 'card2_2', 'forward', 'ebill', 'tollten']
print(model_param)

# MODEL EQUATION
model_eqn = 'ln_total_spend ~ ' + ' + '.join(model_param)
model_eqn

# BUILDING THE MODEL
import statsmodels.formula.api as smf

model0 = smf.ols(model_eqn, credit_data_train).fit()
print(model0.summary())

# FEATURES TO BE USED FOR MODEL BUILDING
model_param = list(set(model_param).difference(['news', 'spousedcat', 'othdebt', 'wireten', 'ownvcr']))
print(model_param)

# MODEL EQUATION
model_eqn = 'ln_total_spend ~ ' + ' + '.join(model_param)
model_eqn

# BUILDING MODEL 1 AFTER REMOVING FEATURES HAVING P-VALUE > 0.9
model1 = smf.ols(model_eqn, credit_data_train).fit()
print(model1.summary())

# FEATURES TO BE USED FOR MODEL BUILDING
model_param = list(set(model_param).difference(['carown', 'jobcat_2', 'employ', 'lncreddebt', 'bfast_2', 'confer', 'homeown', 'callid', 'inccat', 'lnlongten', 'ownipod', 'bfast_3']))
print(model_param)

# MODEL EQUATION
model_eqn = 'ln_total_spend ~ ' + ' + '.join(model_param)
model_eqn

# BUILDING MODEL 2 AFTER REMOVING FEATURES HAVING P-VALUE > 0.7
model2 = smf.ols(model_eqn, credit_data_train).fit()
print(model2.summary())

# FEATURES TO BE USED FOR MODEL BUILDING
model_param = list(set(model_param).difference(['lnequipten', 'vote', 'addresscat', 'wireless', 'polcontrib', 'owncd', 'longten', 'voice', 'hometype_2', 'tollfree', 'jobcat_3', 'owngame', 'forward', 'callwait', 'ownpda', 'tollten', 'lncardten']))
print(model_param)

# MODEL EQUATION
model_eqn = 'ln_total_spend ~ ' + ' + '.join(model_param)
model_eqn

# BUILDING MODEL 3 AFTER REMOVING FEATURES HAVING P-VALUE > 0.4
model3 = smf.ols(model_eqn, credit_data_train).fit()
print(model3.summary())

# FEATURES TO BE USED FOR MODEL BUILDING
model_param = list(set(model_param).difference(['creddebt', 'owndvd', '', 'multline', 'pager', 'owntv', 'retire', 'ownfax']))
print(model_param)

# MODEL EQUATION
model_eqn = 'ln_total_spend ~ ' + ' + '.join(model_param)
model_eqn

# BUILDING MODEL 4 AFTER REMOVING FEATURES HAVING P-VALUE > 0.1
model4 = smf.ols(model_eqn, credit_data_train).fit()
print(model4.summary())

# FEATURES TO BE USED FOR MODEL BUILDING
model_param = list(set(model_param).difference(['jobsat', 'carvalue', 'response_03', 'ownpc', 'hourstv']))
print(model_param)

# MODEL EQUATION
model_eqn = 'ln_total_spend ~ ' + ' + '.join(model_param)
model_eqn

# FINAL MODEL

# BUILDING MODEL 5 AFTER REMOVING FEATURES HAVING P-VALUE > 0.1
model5 = smf.ols(model_eqn, credit_data_train).fit()
print(model5.summary())

# CHECKING THE DISTRIBUTION OF RESIDUALS
model5.resid.mean(), model5.resid.skew()

sns.displot(model5.resid)
plt.show()

# SINCE THE MEAN OF RESIDUALS IS VERY CLOSE TO ZERO AND ALSO FOLLOWS A NORMAL DISTRIBUTION, HENCE WE CAN PROCEED WITH 
# THIS MODEL FOR FURTHER PROCESSING.


# APPLYING THE MODEL ON TRAINING DATASET
credit_data_train.loc[ :, ['total_spend_pred']] = np.exp(model5.predict(credit_data_train))

# APPLYING THE MODEL ON TESTING DATASET
credit_data_test.loc[ :, ['total_spend_pred']] = np.exp(model5.predict(credit_data_test))

# MODEL EVALUATION METRICS

# CORRELATION MATRIX

sns.heatmap(credit_data_train[['total_spend', 'total_spend_pred']].corr(), annot = True)
plt.show()
sns.heatmap(credit_data_test[['total_spend', 'total_spend_pred']].corr(), annot = True)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# MEAN ABSOLUTE ERROR
MAE_train = round(mean_absolute_error(credit_data_train['total_spend'], credit_data_train['total_spend_pred']), 4)
MAE_test = round(mean_absolute_error(credit_data_test['total_spend'], credit_data_test['total_spend_pred']), 4)
print('MAE FOR TRAINING SET : ',  MAE_train , ' | MAE FOR TESTING SET : ', MAE_test)

# MEAN SQUARED ERROR
MSE_train = round(mean_squared_error(credit_data_train['total_spend'], credit_data_train['total_spend_pred']), 4)
MSE_test = round(mean_squared_error(credit_data_test['total_spend'], credit_data_test['total_spend_pred']), 4)
print('MSE FOR TRAINING SET : ',  MSE_train , ' | MSE FOR TESTING SET : ', MSE_test)

# ROOT MEAN SQUARED ERROR
RMSE_train = round(np.sqrt(MSE_train), 4)
RMSE_test = round(np.sqrt(MSE_test), 4)
print('RMSE FOR TRAINING SET : ',  RMSE_train , ' | RMSE FOR TESTING SET : ', RMSE_test)

# MEAN ABSOLUTE PERCENTAGE ERROR
MAPE_train = np.mean(np.abs(credit_data_train['total_spend'] - credit_data_train['total_spend_pred'])/credit_data_train['total_spend'])
MAPE_test  = np.mean(np.abs(credit_data_test['total_spend'] - credit_data_test['total_spend_pred'])/credit_data_test['total_spend'])
print('MAPE FOR TRAINING SET : ',  round(MAPE_train*100, 4) , ' | MAPE FOR TESTING SET : ', round(MAPE_test*100, 4))

# DECILE ANALYSIS
credit_data_train.loc[ :, 'decile'] = pd.qcut(credit_data_train.loc[ :, 'total_spend_pred'], q = 10, labels = False)
credit_data_train.loc[ :, 'decile_count'] = pd.qcut(credit_data_train.loc[ :, 'total_spend_pred'], q = 10, labels = False)
credit_data_test.loc[ :, 'decile']  = pd.qcut(credit_data_test.loc[ :, 'total_spend_pred'], q = 10, labels = False)
credit_data_test.loc[ :, 'decile_count'] = pd.qcut(credit_data_test.loc[ :, 'total_spend_pred'], q = 10, labels = False)

decile_analysis_train = credit_data_train.groupby('decile')[['decile_count', 'total_spend', 'total_spend_pred']].agg({'decile_count' : 'count', 'total_spend' : 'mean', 'total_spend_pred' : 'mean'}).reset_index()
decile_analysis_train.to_csv(path + 'decile_analysis_train.csv')
decile_analysis_train

decile_analysis_test = credit_data_test.groupby('decile')[['decile_count', 'total_spend', 'total_spend_pred']].agg({'decile_count' : 'count', 'total_spend' : 'mean', 'total_spend_pred' : 'mean'}).reset_index()
decile_analysis_test.to_csv(path + 'decile_analysis_test.csv')
decile_analysis_test
