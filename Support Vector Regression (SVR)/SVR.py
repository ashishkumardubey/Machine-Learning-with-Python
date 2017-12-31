# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:21:47 2017

@author: Ashish Dubey
"""

#Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Creating feature matrix and dependent variable martix

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
X
y= dataset.iloc[:,2].values
y


"""#Splitting the dataset into training and test datsets

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state=0)"""


#feature Scaling
#here in SVR feature scaling is required to be done manually since SVM does not do it by its own

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()
sc_y=StandardScaler()

X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)
#Fitting the SVR Model

from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)


#prediciting the results
#We need to scale the inut 6.5 as well as inverse transform it to get actual value without scaling
#also  sc_X need an array as input so numpy array is required to convert int vector to array
# We use sc_X to do feature scaling on matrix of features i.e. 6.5 here  &
# We use sc_y to inverse transfrom the reult from sclaed value to original value
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


# Visualise the result

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Truth or Bluff(SVR)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
