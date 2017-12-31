# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:26:26 2017

@author: Ashish Dubey
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:39:35 2017

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



#Fitting the RANDOM FOREST  Model

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)


#prediciting the results
#We need to scale the inut 6.5 as well as inverse transform it to get actual value without scaling
#also  sc_X need an array as input so numpy array is required to convert int vector to array
# We use sc_X to do feature scaling on matrix of features i.e. 6.5 here  &
# We use sc_y to inverse transfrom the reult from sclaed value to original value
y_pred=regressor.predict(6.5)


# Visualise the result
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Truth or Bluff(Random Forest)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
