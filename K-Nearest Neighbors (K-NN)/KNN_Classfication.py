# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:04:25 2017

@author: Ashish Dubey
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values


# Test and Train split of Datasets

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)



#Feature Scaling of Matrix of features

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

# Fitting the K_NM Model

from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

# Predicting the new Results with Test set data

y_pred=classifier.predict(X_test)

# Creating Confusion Matrix for test set

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

# A great model with only a total of 7 wrong predictions as shown in confusion matrix


#Visualising the training set results


from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step = 0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,
                                           cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X1.min(),X1.max())
 
for i,j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set ==j,0],X_set[y_set== j,1],
     c= ListedColormap(('red','green'))(i),label=j)
plt.title('K-NM Classifier(Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualising the Test set results


from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step = 0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,
                                           cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X1.min(),X1.max())
 
for i,j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set ==j,0],X_set[y_set== j,1],
     c= ListedColormap(('red','green'))(i),label=j)
plt.title('K-NM Classifier(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()