# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:49:11 2017

@author: Ashish Dubey
"""

# Imporitng the libraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv('Mall_Customers.csv')

X= dataset.iloc[:,[3,4]].values


# Using the dendogram to find the optimal numbey of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward')) # ward method is used to minimise the variance within clusters

# The largest line without crossing any horizontal line is the blue one on the rghy most and 
# to finf the number of clusters we count the no of vertical line which is 5 from the dendrogram

# Fitting the Hierarchical clustering to the dataset

from sklearn.cluster import AgglomerativeClustering  # AGGLOMERATIVE means bottom up approach 
hrcluster=AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage ="ward")
y_hrcluster=hrcluster.fit_predict(X)

"""# Renaming a column name and adding a column
dataset[5]= y_hrcluster

dataset.columns.values[5]= 'Clusters'  """

# Visualising the Clusters

plt.scatter(X[y_hrcluster==0,0],X[y_hrcluster==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_hrcluster==1,0],X[y_hrcluster==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_hrcluster==2,0],X[y_hrcluster==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_hrcluster==3,0],X[y_hrcluster==3,1],s=100,c='orange',label='Cluster 4')
plt.scatter(X[y_hrcluster==4,0],X[y_hrcluster==4,1],s=100,c='grey',label='Cluster 5')
plt.title("Clusters of Cients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()



