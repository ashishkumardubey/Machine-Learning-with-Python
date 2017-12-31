# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:46:35 2017

@author: Ashish Dubey
"""
# Importing the important libraries
import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

# Creating the matrix of features and dependent variable vector
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


# Splitting of dataset into training and test is not required as the dataset is small
# Also we want the predicitions to be very accurate and hence will not split this dataset


# Fitting the linear Model
from sklearn.linear_model import LinearRegression
lin_regressor=LinearRegression()
lin_regressor.fit(X,y)


# fitting the ploynomial model
from sklearn.preprocessing import PolynomialFeatures
#We give attribute degree to PolynomialFeatures to create polynomial varibales
#We will use the default degree of 2
ply_regressor=PolynomialFeatures(degree = 4)
X_poly=ply_regressor.fit_transform(X)


# In X_poly the first column is nothing but the constant in the euqation y= b0+b1X1+b2X1squqre+....


# Fitting the model usng Linear Regression

lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)



# Visualising the results for Linear regression

plt.scatter(X,y,color="Red")
plt.plot(X,lin_regressor.predict(X),color="Blue")
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()

# not a great model


# Visualising the results for Polynomial Regression Model
plt.scatter(X,y,color="Red")
plt.plot(X,lin_reg.predict(ply_regressor.fit_transform(X)),color="Blue")
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()





