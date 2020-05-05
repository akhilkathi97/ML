# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart.csv")
#data.profile_report()

train,test = train_test_split(data,test_size=0.33)
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier as dtc
classifier = dtc()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#Random Forest 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)


from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test,y_pred)


import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Heart UCI")
