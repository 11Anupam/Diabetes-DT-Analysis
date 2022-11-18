# -*- coding: utf-8 -*-

#importing all the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#dataframe
df = pd.read_csv("Diabetes.csv")

#decription of numeric data
df.info
df.describe()

# =============================================================================
# EDA:
# =============================================================================
import seaborn as sns
#countplot
sns.countplot(df[' Class variable'])

df.columns
corr=df.corr()
#heatmap:
sns.heatmap(corr,cmap='mako')

#now converting the output carible to binary form:
from sklearn.preprocessing import LabelEncoder        
le=LabelEncoder()
df[' Class variable']=le.fit_transform(df[' Class variable'])

# setting predictors X and target Y:
cols=df.columns
    
X = cols[0:8]
y= cols[8]

# =============================================================================
# Splitting into train test:
# =============================================================================
from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size=0.30)

# =============================================================================
# DT model:
# =============================================================================
from sklearn.tree import DecisionTreeClassifier as DT

model1=DT(criterion='gini')
model1.fit(train[X],train[y])

#predictions:
pred1=model1.predict(test[X])    
pd.crosstab(test[y],pred1,rownames=['Actual'], colnames=['Predictions'])

np.mean(pred1 == test[y])
#70% accouracy

#prediction on training datsaet:
    
pred2=model1.predict(train[X])
pd.crosstab( train[y], pred2, rownames=['actual'],colnames=['predictions'])
np.mean(train[y]==pred2)
#100%

# =============================================================================
#  RandomizedSearchCV
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV

model = DT(criterion = 'entropy')

param_dist = {'min_samples_leaf': list(range(1, 50)),'max_depth': list(range(2, 20)),'max_features': ['sqrt']}
#number of iterations:
n_iter = 50

model_random_search = RandomizedSearchCV(estimator = model, param_distributions = param_dist,n_iter = n_iter)

model_random_search.fit(train[X], train[y])

model_random_search.best_params_

dT_random = model_random_search.best_estimator_

#prediciton on test df 
pred_random = dT_random.predict(test[X])
pd.crosstab(test[y], pred_random, rownames=['Actual'], colnames=['Predictions'])

np.mean(pred_random == test[y])
#74%
#predicition on train df 
pred_random = dT_random.predict(train[X])
pd.crosstab(train[y], pred_random, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(pred_random == train[y])
#76%
