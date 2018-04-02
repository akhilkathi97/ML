# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:16:19 2018

@author: akhil
"""

#import statements
import pandas
import numpy
from sklearn.linear_model import LogisticRegression

#read files
file=pandas.read_csv('diabetes.csv')

#train test split
from sklearn.model_selection import train_test_split
train,test=train_test_split(file,test_size=.32,shuffle=True)

#train test split with input and output separate
train_data=numpy.asarray(train.drop('Outcome',1))
test_data=numpy.asarray(test.iloc[:,:-1])
train_label=numpy.asarray(train['Outcome'])
test_label=numpy.asarray(test.iloc[:,-1])

#normalise 
means=numpy.mean(train_data,axis=0)
stds=numpy.std(train_data,axis=0)
tr_data=(train_data-means)/stds
#meanst=numpy.mean(test_data,axis=0)
#stdst=numpy.std(test_data,axis=0)
ts_data=(test_data-means)/stds


diabetes=LogisticRegression().fit(tr_data,train_label)
accuracy=diabetes.score(ts_data,test_label)

