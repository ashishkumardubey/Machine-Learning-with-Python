# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:06:16 2017

@author: Ashish Dubey
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementing thompson sampling model

import random
N=10000
d=10
ads_selected=[]
numbers_of_rewards_1=[0]*d
numbers_of_rewards_0=[0]*d
total_rewards=0

for n in range(0,N):
    ad=0
    max_random=0    
    for i in range(0,d):
        random_beta= random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
            
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward ==1:
        numbers_of_rewards_1[ad]=numbers_of_rewards_1[ad]+1
    else:
        numbers_of_rewards_0[ad]=numbers_of_rewards_0[ad]+1
    total_rewards=total_rewards+reward
    
# Visualising the results
    
plt.hist(ads_selected)
plt.title('histogram of Ads')
plt.xlabel('Ads')
plt.ylabel('Numbers of times each ad was selected')
plt.show()
