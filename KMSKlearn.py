#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:57:33 2017

@author: fractaluser
"""
from sklearn.datasets import load_iris

iris = load_iris()
type(iris)

## print iris data
iris.data

## print feature names

iris.feature_names

## what we need to predict

iris.target

## encoding 

iris.target_names

## check shape
iris.data.shape

## store feature matrix in X

X = iris.data

## store response in y

y = iris.target

## KNN model - Classification

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn #check default values for other parameters

## fit the model with data
knn.fit(X, y)

## make prediction

knn.predict([3, 5, 4, 2])

## multiple predictions

X_new = [[3,5, 4, 2], [5, 4, 3, 2]]

knn.predict(X_new)

## changing value of parameters

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X, y)

knn.predict(X_new)


## Using different classification model

from sklearn.linear_model import LogisticRegression

## instantiate the model

logreg = LogisticRegression()

logreg.fit(X, y)

logreg.predict(X_new)

# TRAIN and TEST on entire dataset

logreg.fit(X, y)

## predict for observations in X
y_pred = logreg.predict(X)


## classification accuracy

from sklearn import metrics
metrics.accuracy_score(y, y_pred) #training accuracy

## try with knn

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)

y_pred = knn.predict(X)

metrics.accuracy_score(y, y_pred)

## Using train-test split in data

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)

## STEP 2: train model on training set

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

## STEP 3: Make predictions on the testing set

y_pred = logreg.predict(X_test)

## Compare responses, check accuracy

metrics.accuracy_score(y_test, y_pred)

## Repeat with KNN

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


## try every value of K using a for loop

k_range = range(1, 26)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
## import matplotlib (plotting library)

import matplotlib.pyplot as plt

## plot relationship b/w K and tesing accuracy

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


## Optimum parameters

knn = KNeighborsClassifier(n_neighbors = 11)

knn.fit(X, y)
knn.predict([3,5,4,2]) #making predictions

