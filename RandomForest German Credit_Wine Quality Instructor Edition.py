#!/usr/bin/env python
# coding: utf-8

# ## Chapter 9 Random Forest Classifier and Regressor
# 
# ### Classifier Data = German Bank Credit Risk
# ### Regressor Data = Portuguese Wine Quality

# ### German Credit Dataset<br>
# 
# This dataset comes from a German bank and it's used to determine if a customer is a good credit risk or a bad credit risk (RESPONSE = 1 or 0, respectively). It's better to grade a good customer as bad (false positive) than to grade a bad customer as good (false negative), so keep this in mind when reviewing the confusion matrices.<br>
# 
# Remember, good credit risk is a 1 and bad credit risk is a zero.<br>

# ### Data Dictionary:
# 
# Attribute 1: (qualitative)<br>
# Status of existing checking account<br>
# A11 : ... < 0 DM<br>
# A12 : 0 <= ... < 200 DM<br>
# A13 : ... >= 200 DM / salary assignments for at least 1 year<br>
# A14 : no checking account<br>
# 
# Attribute 2: (numerical)<br>
# Duration in month<br>
# 
# Attribute 3: (qualitative)<br>
# Credit history<br>
# A30 : no credits taken/ all credits paid back duly<br>
# A31 : all credits at this bank paid back duly<br>
# A32 : existing credits paid back duly till now<br>
# A33 : delay in paying off in the past<br>
# A34 : critical account/ other credits existing (not at this bank)<br>
# 
# Attribute 4: (qualitative)<br>
# Purpose<br>
# A40 : car (new)<br>
# A41 : car (used)<br>
# A42 : furniture/equipment<br>
# A43 : radio/television<br>
# A44 : domestic appliances<br>
# A45 : repairs<br>
# A46 : education<br>
# A47 : (vacation - does not exist?)<br>
# A48 : retraining<br>
# A49 : business<br>
# A410 : others<br>
# 
# Attribute 5: (numerical)<br>
# Credit amount<br>
# 
# Attibute 6: (qualitative)<br>
# Savings account/bonds<br>
# A61 : ... < 100 DM<br>
# A62 : 100 <= ... < 500 DM<br>
# A63 : 500 <= ... < 1000 DM<br>
# A64 : .. >= 1000 DM<br>
# A65 : unknown/ no savings account<br>
# 
# Attribute 7: (qualitative)<br>
# Present employment since<br>
# A71 : unemployed<br>
# A72 : ... < 1 year<br>
# A73 : 1 <= ... < 4 years<br>
# A74 : 4 <= ... < 7 years<br>
# A75 : .. >= 7 years<br>
# 
# Attribute 8: (numerical)<br>
# Installment rate in percentage of disposable income<br>
# 
# Attribute 9: (qualitative)<br>
# Personal status and sex<br>
# A91 : male : divorced/separated<br>
# A92 : female : divorced/separated/married<br>
# A93 : male : single<br>
# A94 : male : married/widowed<br>
# A95 : female : single<br>
# 
# Attribute 10: (qualitative)<br>
# Other debtors / guarantors<br>
# A101 : none<br>
# A102 : co-applicant<br>
# A103 : guarantor<br>
# 
# Attribute 11: (numerical)<br>
# Present residence since<br>
# 
# Attribute 12: (qualitative)<br>
# Property<br>
# A121 : real estate<br>
# A122 : if not A121 : building society savings agreement/ life insurance<br>
# A123 : if not A121/A122 : car or other, not in attribute 6<br>
# A124 : unknown / no property<br>
# 
# Attribute 13: (numerical)<br>
# Age in years<br>
# 
# Attribute 14: (qualitative)<br>
# Other installment plans<br>
# A141 : bank<br>
# A142 : stores<br>
# A143 : none<br>
# 
# Attribute 15: (qualitative)<br>
# Housing<br>
# A151 : rent<br>
# A152 : own<br>
# A153 : for free<br>
# 
# Attribute 16: (numerical)<br>
# Number of existing credits at this bank<br>
# 
# Attribute 17: (qualitative)<br>
# Job
# A171 : unemployed/ unskilled - non-resident<br>
# A172 : unskilled - resident<br>
# A173 : skilled employee / official<br>
# A174 : management/ self-employed/<br>
# highly qualified employee/ officer<br>
# 
# Attribute 18: (numerical)<br>
# Number of people being liable to provide maintenance for<br>
# 
# Attribute 19: (qualitative)<br>
# Telephone<br>
# A191 : none<br>
# A192 : yes, registered under the customers name<br>
# 
# Attribute 20: (qualitative)<br>
# foreign worker<br>
# A201 : yes<br>
# A202 : no<br>

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pylab as plt

from dmba import classificationSummary, regressionSummary, gainsChart, liftChart


# In[2]:


data_df = pd.read_csv('GermanCredit.csv')


# In[9]:


data_df.head()


# In[3]:


# drop unncessary predictors

data_df = data_df.drop(columns=['OBS#'])


# In[12]:


# check the number of samples and variables

data_df.shape


# In[13]:


# inspect the variable names, non-null values, and datatypes

data_df.info()


# In[4]:


# Create X and y objects

X = data_df.drop(columns=['RESPONSE'])

y = data_df.RESPONSE


# In[5]:


# Convert the y response variable to categorical and back to integer values

y = y.astype('category').cat.codes


# In[6]:


# display the class membership of the response variable (counts)

y.value_counts()


# In[7]:


# Split the data into training and test sets with 60% for training and 40% for test; preserve class ratios

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)


# #### Build a baseline logistic regression model<br>
# 
# 1) Set ridge regression penalty<br>
# 2) Search 100 values of lambda<br>
# 3) Set 10-fold cross validation<br>
# 4) Use the liblinear solver<br>
# 5) Set class weight to balanced<br>
# 6) Use accuracy as the scoring measure
# 7) Start with 1,000 iterations and increase as necessary<br>

# In[19]:


# Build a logistic regression model as a baseline

logit_reg = LogisticRegressionCV(penalty="l2", Cs=100, solver='liblinear', cv=10,
                                 class_weight='balanced', scoring='accuracy', max_iter=1000)

logit_reg.fit(train_X, train_y)


# In[20]:


# display confusion matrices for train and test data

classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(test_y, logit_reg.predict(test_X))


# In[21]:


# display classification report for the test data

classes = logit_reg.predict(test_X)

print(metrics.classification_report(test_y, classes))


# ### Build a default RandomForest classifier

# In[22]:


# Rerun the same train/test split as before

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)


# 1) Build a standard RandomForest model<br>
# 1) Set 500 as the number of trees to be built<br>
# 2) Set the random seed value to 1<br>

# In[23]:


rf = RandomForestClassifier(n_estimators=500, random_state=1)

rf.fit(train_X, train_y)


# In[24]:


# display confusion matrices for train and test data

classificationSummary(train_y, rf.predict(train_X))
classificationSummary(test_y, rf.predict(test_X))


# In[ ]:


# display classification report for the test data

classes = rf.predict(test_X)

print(metrics.classification_report(test_y, classes))


# ### Build a hyperparameter-tuned RandomForest model

# In[6]:


# Rerun the same train/test split as before

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)


# #### Use the GridSearchCV algorithm to search hyperparameter tuning values<br>
# ##### Set the parameter grid as follows:<br>
# 
# 1) Try 300, 400, 500, 700, and 1000 estimators<br>
# 2) Use both gini and entropy<br>
# 3) Set OOB score to true<br>
# 4) Check minimum impurity decrease of 0.0001, 0.0005, 0.005, 0.001<br>
# 5) Check minimum samples per split of 2, 4, 6, 8, 10<br>
# 6) Set class weight to balanced.<br>
# 7) Set random seed value to 1<br>

# In[12]:


# user grid search to find optimized tree
param_grid = {
    'n_estimators': [300, 500, 700], 
    'criterion' : ['gini'],
    'oob_score': [True],
    'min_impurity_decrease': [0.0001, 0.005, 0.001], 
    'min_samples_split': [2, 6, 10], 
    'class_weight':['balanced'],
    'random_state':[1],
}


# #### Note: The cv parameter has been set to its lowest level (cv=2); RandomForest doesn't require cross-validation but it does need grid search for hyperparameter selection and GridSearchCV doesn't allow disabling the CV functionality.

# In[13]:


gridSearch = GridSearchCV(RandomForestClassifier(), param_grid, cv=2, n_jobs=6)

gridSearch.fit(train_X, train_y)

print('Initial parameters: ', gridSearch.best_params_)

rfTree = gridSearch.best_estimator_


# In[14]:


# Display the Out-Of-Bag accuracy score

rfTree.oob_score_


# In[15]:


# display confusion matrices for train and test data

classificationSummary(train_y, rfTree.predict(train_X))
classificationSummary(test_y, rfTree.predict(test_X))


# In[57]:


# display classification report for the test data

classes = rf.predict(test_X)

print(metrics.classification_report(test_y, classes))


# In[42]:


# Display the feature importances

get_ipython().run_line_magic('matplotlib', 'inline')

train_X = pd.DataFrame(train_X)

importances = rfTree.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfTree.estimators_], axis=0)

df = pd.DataFrame({'feature': train_X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance')
print(df)

ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')

plt.tight_layout()
plt.show()


# In[51]:


# Display the lift and gain charts

rfTree_pred = rfTree.predict(test_X)
rfTree_proba = rfTree.predict_proba(test_X)
rfTree_result = pd.DataFrame({'actual': test_y, 
                             'p(0)': [p[0] for p in rfTree_proba],
                             'p(1)': [p[1] for p in rfTree_proba],
                             'predicted': rfTree_pred })

df = rfTree_result.sort_values(by=['p(1)'], ascending=False)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

gainsChart(df.actual, ax=axes[0])
liftChart(df['p(1)'], title=False, ax=axes[1])
    
plt.tight_layout()
plt.show()


# In[52]:


# Display the ROC chart

rfTree_pred = rfTree.predict(test_X)
rfTree_proba = rfTree.predict_proba(test_X)

preds = rfTree_proba[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_y, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('RFTree_ROC')
plt.show()


# ### Explainable Boosting Machine

# In[9]:


from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier()
ebm.fit(train_X, train_y)


# In[12]:


# display confusion matrices for train and test data

classificationSummary(train_y, ebm.predict(train_X))
classificationSummary(test_y, ebm.predict(test_X))


# In[10]:


from interpret import show

ebm_global = ebm.explain_global()
show(ebm_global)


# In[ ]:


ebm_local = ebm.explain_local(test_X, test_y)
show(ebm_local)


# ### RandomForest Regression Model

# There are two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# This first modeling exercise is a regression using 'quality' as a continuous numeric value (even though it is integer).

# In[13]:


wine_df = pd.read_csv("winequality-white.csv", sep = ";")


# In[68]:


# Inspect the first five rows

wine_df.head()


# In[69]:


# Display the descriptive statistics

wine_df.describe()


# In[82]:


# Display the variable names, non-null values, and datatypes

wine_df.info()


# In[14]:


# Replace spaces in variable names with underscores

wine_df.columns = [s.strip().replace(' ', '_') for s in wine_df.columns]
wine_df.columns


# In[15]:


# quality is the response variable and everything else is a predictor (make X and y here)

y = wine_df['quality']

X = wine_df.drop(columns=['quality'], inplace = False)


# In[16]:


# Convert the response variable from integer to float for better regression performance

y = y.astype('float')


# In[17]:


# split the data into training and test, with 60% for train and 40% for test

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.4, random_state=1)


# ### Random Forest Regression Model Here

# In[86]:


# user grid search to find optimized tree (this should be a iterative process)

param_grid = {
    'n_estimators': [300, 500, 700], 
    'min_impurity_decrease': [0.0001, .0005, 0.001], 
    'min_samples_split': [10, 20, 30], 
}


# In[87]:


# because RandomForest already has two-way crossvalidation, we set the GridSearchCV "cv" to its minimum value

gridSearch = GridSearchCV(RandomForestRegressor(), param_grid, cv=2, n_jobs=6)

gridSearch.fit(train_X, train_y)


# In[88]:


# Display the initial parameters and put the best model into an object named "rfTree"

print('Initial parameters: ', gridSearch.best_params_)

rfTree = gridSearch.best_estimator_


# In[89]:


# Display the error performance measures

regressionSummary(train_y, rfTree.predict(train_X))
print()
regressionSummary(test_y, rfTree.predict(test_X))


# In[79]:


# Display the feature importances

get_ipython().run_line_magic('matplotlib', 'inline')

importances = rfTree.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfTree.estimators_], axis=0)

df = pd.DataFrame({'feature': train_X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance')
print(df)

ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')

plt.tight_layout()
plt.show()


# In[81]:


# Display the regression gain and lift charts

pred_v = pd.Series(rfTree.predict(test_X))
pred_v = pred_v.sort_values(ascending=False)

fig,axes = plt.subplots(nrows=1, ncols=2)
ax = gainsChart(pred_v, ax=axes[0])
ax.set_ylabel('Cumulative Area Prediction')
ax.set_title("Cumulative Gains Chart")

ax = liftChart(pred_v, ax=axes[1], labelBars=False)
ax.set_ylabel("Lift")

plt.tight_layout()
plt.show()


# ### Explainable Boosting Machine Regression

# In[22]:


from interpret.glassbox import ExplainableBoostingRegressor

ebm = ExplainableBoostingRegressor()
ebm.fit(train_X, train_y)


# In[23]:


# Display the error performance measures

regressionSummary(train_y, ebm.predict(train_X))
print()
regressionSummary(test_y, ebm.predict(test_X))


# In[19]:


from interpret import show

ebm_global = ebm.explain_global()
show(ebm_global)


# In[ ]:




