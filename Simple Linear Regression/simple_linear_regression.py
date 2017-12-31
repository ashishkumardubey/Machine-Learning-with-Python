# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:47:56 2017

@author: Ashish Dubey
"""

#Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Creating feature matrix and dependent variable martix

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
X
y= dataset.iloc[:,1].values
y


#Splitting the dataset into training and test datsets

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state=0)

# Fitting the regression Model

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X_train,y_train)


#predict the test set values

y_pred=regressor.predict(X_test)


#Visualising the  y_test vs y_pred i.e. actual values verses predicted values by the model


plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Salary vs years of experience')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()


