# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
testfile = pd.read_csv('train.csv')
#X=testfile.iloc[:,:-1].x 
#Y=testfile.iloc[:,1:].y
X=testfile['x']
X.values.reshape(len(X),1)
Y=testfile['y']
Y.values.reshape(len(Y),1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.25,shuffle=True)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train,Y_train)
