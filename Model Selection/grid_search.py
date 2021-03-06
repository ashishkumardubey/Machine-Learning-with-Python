# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:53:52 2017

@author: Ashish Dubey
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

from sklearn.svm import SVC
classifier=SVC(kernel = 'rbf', random_state=0)
classifier.fit(X_train,y_train)
classifier.score(X_train,y_train)
# Predicting the new Results with Test set data

y_pred=classifier.predict(X_test)

# Creating Confusion Matrix for test set

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

# Applying K-Fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier,X=X_train, y = y_train, cv=10)
accuracies.mean()
accuracies.std()

# Applying Grid _Search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                          )
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

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
plt.title('SVM(Training Set)')
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
plt.title('SVM(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()