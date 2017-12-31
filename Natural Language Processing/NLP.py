# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:50:56 2017

@author: Ashish Dubey
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t', quoting=3)


# Cleaning the dataset

import re
import nltk
#  to download stopwords   
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in  review if not word in set(stopwords.words('english'))] # Using set will increase the algorithm processing speed for longer text records.
    
    # Joining the list of words together
    review= ' '.join(review) # we add ' ' to add space between words or else all words will be joined to gether to be one single meaningless word
    corpus.append(review)
    
    
# Creating  the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray() # X is a Sparse matrix

#Dependent Variable

y=dataset.iloc[:,1].values



# we will use classification model to build model , naive bayes, decision tree & random forest are most commonly used

# Test and Train split of Datasets

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)



#Feature Scaling of Matrix of features

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

# Fitting the NAIVE Bayes Model

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the new Results with Test set data

y_pred=classifier.predict(X_test)

# Creating Confusion Matrix for test set

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

