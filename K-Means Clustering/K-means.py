# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:15:33 2017

@author: Ashish Dubey
"""

# %reset -f to reser everything


# Imporitng the libraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv('Mall_Customers.csv')

X= dataset.iloc[:,[3,4]].values


# Using the elbow method to find optimal numbers of clusters

from sklearn.cluster import KMeans
wcss= []
for i in range(1,11):
    kmeans=KMeans(n_clusters= i, init ='k-means++',max_iter= 300,n_init =10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
    plt.plot(range(1,11), wcss)
    plt.title('the elbow method')
    plt.xlabel('No of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    
    
#Applying k0means to the dataset

kmeans=KMeans(n_clusters= 5, init ='k-means++',max_iter= 300,n_init =10,random_state=0)
y_kmeans =kmeans.fit_predict(X)



#Visualising the Clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='pink',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='grey',label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='Centroid')
plt.title("Clusters of Cients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()