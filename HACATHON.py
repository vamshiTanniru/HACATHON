#!/usr/bin/env python
# coding: utf-8

# In[810]:


# Import the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# for HD visualizations
get_ipython().run_line_magic('config', "InlinrBackend.figure_format ='retina'")


# In[886]:


# Loading csv file
df = pd.read_csv(r"C:\Users\Vamsh\Downloads\dataframe_.csv")


# In[812]:


df.head()


# In[813]:


df.shape


# In[814]:


df.info()


# In[815]:


df.isna().value_counts()


# In[816]:


df[df.output.isna()]


# In[817]:


df[df.input.isna()]


# In[818]:


mean=round(df.input.mean(),2)


# In[819]:


mean1=round(df.output.mean(),2)


# In[820]:


# Replacing Null values with Mean

df.input.replace(np.nan,mean,inplace=True)

df.output.replace(np.nan,mean1,inplace=True)


# In[821]:


# NUll values is Replaced with Mean values

df.isna().sum()


# In[822]:


# check the duplicates

df.duplicated()
df.duplicated().value_counts()


# In[823]:


df.describe()


# In[824]:


# remove duplicates

df = df.drop_duplicates()


# In[825]:


df
df.duplicated().value_counts()


# In[826]:


# Checking the skewsness
df.skew()


# In[827]:


# Identify the outliers

Q1,Q2,Q3 = tuple(df.output.quantile(q = [0.25, 0.5, 0.75 ]).values)


# In[828]:


print(Q1, Q2, Q3)


# In[829]:


IQR = Q3-Q1
UL = Q3 + 1.5*IQR
LL = Q1 - 1.5*IQR

print(IQR, UL, LL)


# In[830]:


(df[(df.output > UL) | (df.output < LL)]).count()


# ## Machine Learning

# ## Analysis

# In[831]:


sns.boxplot(df.input);


# In[832]:


sns.boxplot(df.output);


# In[833]:


sns.pairplot(df)


# In[834]:


sns.histplot(df.input,kde=True);


# In[835]:


sns.heatmap(df.corr());


# In[836]:


df.columns


# ## a. Identify the Target Variable and Splitting the Data into Train and Test

# In[837]:


# Identify the inputs(X) and output (y)

y = df['output']
X = df[['input']]


# In[838]:


# Split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.75, random_state = 143)


# In[839]:


X_train.head()


# In[840]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ## b. Seperate the Categorial  and Numerical Columns:

# In[841]:


X_train.head()


# In[842]:


X_train.dtypes


# In[843]:


X_train_num = X_train.select_dtypes(include=['float64','int64'])

X_train_num.head()


# ## c. Scaling the Numerical Features

# In[844]:


X_train_num.head()


# In[845]:


# scaling the numeroical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# column names are (annoyingly) lost after Scaling
# (i.e. the dataframe is converted to a numpy ndarray)

X_train_num_rescaled = pd.DataFrame(scaler.fit_transform(X_train_num),
                                   columns = X_train_num.columns,
                                   index = X_train_num.index)

X_train_num_rescaled.head()


# In[846]:


X_train_num.describe()


# In[847]:


print('Number of Numerical Feature:', scaler.n_features_in_)
print('Mean of each column:', scaler.mean_)
print('std of each column:', np.sqrt(scaler.var_))


# ## d. Concatinating the Encoded Categorical Features and Rescaled Numerical Features

# In[848]:


X_train_transformed = pd.concat([X_train_num_rescaled])

X_train_transformed.head()


# ## g. Preparing Test Data

# In[849]:


X_test.head()


# In[850]:


X_test.info()


# In[851]:


X_test_num = X_test.select_dtypes(include=['int64', 'float64'])

X_test_num.head()


# In[852]:


X_test_num_rescaled = pd.DataFrame(scaler.transform(X_test_num), 
                                   columns = X_test_num.columns, 
                                   index = X_test_num.index)

X_test_num_rescaled.head()


# In[853]:


X_test_transformed = pd.concat([X_test_num_rescaled])

X_test_transformed.head()


# ## Linear Regression

# In[854]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_transformed, y_train)


# In[855]:


# Prediction

y_test_pred = regressor.predict(X_test_transformed)


# In[856]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[857]:


sns.histplot(y_test, color='blue', alpha=0.5);
sns.histplot(y_test_pred, color='red', alpha=0.5);


# In[858]:


# Evaluation

from sklearn import metrics

MeanAbsoluteError_LR = metrics.mean_absolute_error(y_test, y_test_pred)
MeanSquaredError_LR = metrics.mean_squared_error(y_test, y_test_pred)
RootMeanSquaredError_LR = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
                                                
print('Mean Absolute Error: ', MeanAbsoluteError_LR)

print('Mean Squared Error: ', MeanSquaredError_LR)

print('Root Mean Squared Error: ', RootMeanSquaredError_LR)


# ## KNN Regression

# In[859]:


from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor()
regressor.fit(X_train_transformed, y_train)


# In[860]:


# Prediction

y_test_pred = regressor.predict(X_test_transformed)


# In[861]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[862]:


sns.histplot(y_test, color='blue', alpha=0.5)
sns.histplot(y_test_pred, color='red', alpha=0.5);


# In[863]:


from sklearn import metrics

MeanAbsoluteError_KNN = metrics.mean_absolute_error(y_test, y_test_pred)
MeanSquaredError_KNN = metrics.mean_squared_error(y_test, y_test_pred)
RootMeanSquaredError_KNN = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
                                                
print('Mean Absolute Error: ', MeanAbsoluteError_KNN)

print('Mean Squared Error: ', MeanSquaredError_KNN)

print('Root Mean Squared Error: ', RootMeanSquaredError_KNN)


# ## support Vector Regression

# In[864]:


from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X_train_transformed, y_train)
y_test_pred = regressor.predict(X_test_transformed)
temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()
sns.histplot(y_test, color='blue', alpha=0.5)
sns.histplot(y_test_pred, color='red', alpha=0.5);



# In[865]:


MeanAbsoluteError_SVR = metrics.mean_absolute_error(y_test, y_test_pred)
MeanSquaredError_SVR = metrics.mean_squared_error(y_test, y_test_pred)
RootMeanSquaredError_SVR = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# ## Decision Tree Regression

# In[866]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train_transformed, y_train)


# In[867]:


y_test_pred = regressor.predict(X_test_transformed)


# In[868]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[869]:


sns.histplot(y_test, color='blue', alpha=0.5)
sns.histplot(y_test_pred, color='red', alpha=0.5);


# In[870]:


from sklearn import metrics

MeanAbsoluteError_DTR = metrics.mean_absolute_error(y_test, y_test_pred)
MeanSquaredError_DTR = metrics.mean_squared_error(y_test, y_test_pred)
RootMeanSquaredError_DTR = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
                                                
print('Mean Absolute Error: ', MeanAbsoluteError_DTR)

print('Mean Squared Error: ', MeanSquaredError_DTR)

print('Root Mean Squared Error: ', RootMeanSquaredError_DTR)


# ## Ensemble
# 
# ## Random Forest Regression

# In[871]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train_transformed, y_train)


# In[872]:


# Prediction

y_test_pred = regressor.predict(X_test_transformed)


# In[873]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[874]:


sns.histplot(y_test, color='blue', alpha=0.5)
sns.histplot(y_test_pred, color='red', alpha=0.5)


# In[875]:


from sklearn import metrics

MeanAbsoluteError_RFR = metrics.mean_absolute_error(y_test, y_test_pred)
MeanSquaredError_RFR = metrics.mean_squared_error(y_test, y_test_pred)
RootMeanSquaredError_RFR = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
                                                
print('Mean Absolute Error: ', MeanAbsoluteError_RFR)

print('Mean Squared Error: ', MeanSquaredError_RFR)

print('Root Mean Squared Error: ', RootMeanSquaredError_RFR)


# ## Boosting
# 
# ## Gradient Boosted Decision Tree

# In[876]:


from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()
regressor.fit(X_train_transformed, y_train)


# In[877]:


# Prediction

y_test_pred = regressor.predict(X_test_transformed)


# In[878]:


temp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})

temp_df.head()


# In[879]:


sns.histplot(y_test, color='blue', alpha=0.5)
sns.histplot(y_test_pred, color='red', alpha=0.5);


# In[880]:


from sklearn import metrics

MeanAbsoluteError_GBDT = metrics.mean_absolute_error(y_test, y_test_pred)
MeanSquaredError_GBDT = metrics.mean_squared_error(y_test, y_test_pred)
RootMeanSquaredError_GBDT = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
                                                
print('Mean Absolute Error: ', MeanAbsoluteError_GBDT)

print('Mean Squared Error: ', MeanSquaredError_GBDT)

print('Root Mean Squared Error: ', RootMeanSquaredError_GBDT)


# In[881]:


df_data = [['Linear Regression',MeanAbsoluteError_LR, MeanSquaredError_LR, RootMeanSquaredError_LR],
                   ['K-Nearest Neighbor Regression', MeanAbsoluteError_KNN, MeanSquaredError_KNN, RootMeanSquaredError_KNN ],
                   ['Decision Tree Regression', MeanAbsoluteError_DTR,MeanSquaredError_DTR, RootMeanSquaredError_DTR ],
                   ['Random Forest Regression ', MeanAbsoluteError_RFR,MeanSquaredError_RFR, RootMeanSquaredError_RFR ],
           ['support Vector Regression',MeanAbsoluteError_SVR,MeanSquaredError_SVR,RootMeanSquaredError_SVR],
                   ['Gradient Boosting Decision Tree', MeanAbsoluteError_GBDT,MeanSquaredError_GBDT, RootMeanSquaredError_GBDT ]]

data = pd.DataFrame(df_data, columns = ['Algorithm','Mean Absolute Error','Mean Square Error','Root Mean Square Error'])


# In[882]:


data


# ## Observation
# 
# 
#  * Linear Regression---Mean Absolute Error:  46.620273
#  * KNN Regression   ---Mean Absolute Error:  22.421405
#  * Decision Tree Regression --- Mean Absolute Error:  29.945462
#  * Random Forest Regression --- Mean Absolute Error:  25.683183
#  * Support Vector Regression --- Mean Absolute Error: 26.960490 	 
#  * Gradient Boosting Regressor --- Mean Absolute Error:  21.756139
#  
# 
# **1. By observing the above table we can say that Gradient Boosting Decision Tree Mean absolute error is less i.e. 25.067372 compaired to other algorithms.**
# 
# **2. By compairing all algorithms we can easily says that Gradient Boosting Decision tree is the best algorithm for the Output Prediction.**

# In[884]:


sns.barplot(data);


# In[885]:


sns.barplot(y=data.Algorithm,x=data['Mean Absolute Error'],ci=False,orient='h');


# ## Conclusion
# 
# **Gradient Boosting Decision tree is the best algorithm for the Output Prediction.**

# In[ ]:




