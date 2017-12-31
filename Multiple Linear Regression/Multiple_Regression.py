# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:53:58 2017

@author: Ashish Dubey
"""

#Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Creating feature matrix and dependent variable martix

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
X
y= dataset.iloc[:,4].values
y


#Encoding the Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#Avoiding the dummy variable trap

X=X[:,1:]


#Splitting the dataset into training and test datsets

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=0)



# Fitting the regression Model

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)



#predict the test set values

y_pred=regressor.predict(X_test)

#Building the Optimal Model using backward elimation
#First need to manually add constabt value of 1 that is x0 so that the equation y=mx+c is validated
#here in multiple linear regression y= b0 + b1x1+b2x2+......bnxn so for b0,x0 is equal to 1
# This x0 value is not by default and needs to be manually added

import statsmodels.formula.api as sm

X=np.append(arr=np.ones((50,1)).astype(int),values = X, axis =1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


#X1 has the highest p-value of 0.484 which is much higher than the Significance Level of 0.05
#We will remove x1 from matrix of features for next iteration 

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#Now X2 that is 2nc column has hghest p-value of 0.397 which is higher than 0.05

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#This is the optimal model which has all the p-values less than 0.05 and all significant values

#Fitting the new best model

#Splitting the dataset into training and test datsets

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_opt,y,test_size = 0.3, random_state=0)



# Fitting the regression Model

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)



#predict the test set values

y_prediction=regressor.predict(X_test)

