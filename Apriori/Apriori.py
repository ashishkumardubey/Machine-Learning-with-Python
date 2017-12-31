# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:00:33 2017

@author: Ashish Dubey
"""
 
# Imporitng the libraries
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt

# Importing the dataset
 dataset = pd.read_csv('Market_Basket_Optimisation.csv',header= None)
 
 # We need to create list of list containing the transactions for the algorithm to run
 # Creating a list of lists
 
 transactions =[]
 for i in range(0,7501):
     transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
     
     
#  Training the Apriori on the Dataset
     
from apyori import apriori

rules = apriori(transactions, min_support=0.003,min_confidence=0.2,min_lift= 3,min_length=2)

# Visualising the results

results = list(rules)
