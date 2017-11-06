#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:57:33 2017

@author: fractaluser
"""
from sklearn.datasets import load_iris
import numpy as np

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




### DS in Python: Pandas, Seaborn, SciKit-Learn ###

# Pandas to read data
# seaborn to visualize data
# sci-kit learn to model data

import pandas as pd

# read csv file directly from a URL
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)

data.head()

data.tail()

data.shape


### Visualizing data using seaborn ###

import seaborn as sns

#allow plots to appear within the notebook

%matplotlib inline

# visualize the relationship between the features and the response variables using  scatterplots
data.columns


sns.pairplot(data, x_vars = ['TV', 'radio', 'newspaper'], y_vars = 'sales', size = 5, aspect = 0.5, kind = 'reg') #scatter plot, line of best fit and 95% confidence interval


# select a subset of the original dataframe
feature_cols = ['TV', 'radio', 'newspaper']

X = data[['TV', 'radio', 'newspaper']]

# check type() and shape of X

y = data['sales']

## Splitting X and y into training and testing sets

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)

# import model
from sklearn.linear_model import LinearRegression

# instantiate
linreg = LinearRegression()

# fit the model to the training data

linreg.fit(X_train, y_train)


## interpreting model coefficients

# Print intercept and coefficients

linreg.intercept_
linreg.coef_


# make predictions on the testing set
y_pred = linreg.predict(X_test)

# evaluating predictions

# MAE (Mean Absolute Error)

from sklearn import metrics
metrics.mean_absolute_error(y_pred, y_test)

# MSE (Mean Squared Error)

metrics.mean_squared_error(y_pred, y_test)

# RMSE (Root Mean Square Error)

np.sqrt(metrics.mean_squared_error(y_pred, y_test)) #squares error, so increases the weight of larger errors


## FEATURE SELECTION

feature_cols = ['TV', 'radio']

# use list for data
X = data[feature_cols]

# split into train and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)

# fit the model
linreg.fit(X_train, y_train)

# make predictions
y_pred = linreg.predict(X_test)

# compute RMSE
np.sqrt(metrics.mean_squared_error(y_pred, y_test)) #lower, thus new model better

# check out seaborn library for visualization




### SELECTING BEST MODEL using CROSS VALIDATION ###

#Train/Test split provides high variance estimate depending on what values are in the testing set

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris = load_iris()

# create X (features) and y (response)

X = iris.data
y = iris.target

# use train/test split with different random state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)

# check classification accuracy of KNN with K-5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

metrics.accuracy_score(y_test, y_pred)

# K-fold Cross Validation
# 1. split data into K equal partitions (folds)
# 2. Use fold 1 as testing set and union on all other folds as training sets
# 3. calculate testing accuracy
# 4. repeat 2 & 3 K times, using a different fold as the testing set each time
# 5. use average testing accuracy as the estimate of out-of-sample accuracy

# Advantages of CV:
# 1. generates more accurate estimate of out-of-sample accuracy
# 2. more efficient use of data (used for both training and testing)

# Advantages of train/test split
# 1. runs k times faster than k-fold cv
# 2. simpler to examine results


## Cross Validation recommendations
# 1. K = 10 generally recommended
# 2. for classification problems, stratified sampling is recommended for creating the folds
    # - each response class should be represented with equal proportions in each of the K-folds
    # - scikit learn's cross_val_score fn does this by default
    
# Using Cross validation for parameter tuning
    
from sklearn.cross_validation import cross_val_score

# 10-fold cross validation for k = 5 for KNN (n = 5)

knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy') #'accuracy': classification accuracy to be used

scores
# returns the 10 accuracy scores as a np array

# use average accuracy
scores.mean()

# search for an optimal value of K for KNN

k_range = range(1,31)

k_scores = []
for k in k_range:
    knn= KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    k_scores.append(scores.mean())
    
k_scores

# Use a line plot to visualize accuracy change
import matplotlib.pyplot as plt

# plot the value K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# example of bias variance trade off
# generally choose the values for simpler models

# with KNN higher values of K produce lower complexity models, therefore use 20 here


# USE CROSS VALIDATION to choose b/w models
# 10-fold cross validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors = 20)
cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy').mean()

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy').mean()


# USE CV for feature selection

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read data in the advertising dataset
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)

# create a Python list of 3 feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of df (X)

X = data[feature_cols]

# select the sales column as the response (y)
y = data['sales']

# 10-fold cross-validation with all three features
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv = 10, scoring = 'mean_squared_error') #use MSE for Linear Regression problems

# fix the sign of the MSE scores
mse_scores = -scores
# higher results better values, so loss functions are always negative in scikit learn

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)

#mean RMSE
rmse_scores.mean()

# 10 fold cv with 2 features (no newspaper)

feature_cols = ['TV', 'radio']
X = data[feature_cols]
y = data['sales']

lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv = 10, scoring = 'mean_squared_error')

rmse_scores = np.sqrt(-scores)
rmse_scores.mean()


## Improvement to cross-validation

# - Repeated cross-validation
# - Create hold out set, a portion not touched during the model building

# - Carry out Feature engineering during each iteration of the cross validation



### BEST MODEL PARAMETERS IN SCIKIT LEARN ###

# K-fold cross validation for an optimal tuning parameter


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

# 10-fold cross validation with K = 5 for KNN (the n-neighbors parameter)
knn = KNeighborsClassifier(n_neighbors = 5)

scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')

scores.mean()

# search for optimal value of K

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    k_scores.append(scores.mean())
    
    
# plot the value of K v/s accuracy
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# automate the above process using GridSearchCV

from sklearn.grid_search import GridSearchCV

# define the parameter values that should be searched
k_range = list(range(1,31))


# create a parameter grid: map the parameter names to the values to be searched
param_grid = dict(n_neighbors = k_range)

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')


# fit the grid with data
grid.fit(X, y)

# view the complete results (list of names tuples)
grid.grid_scores_ #list of 30 names tuples n = 1 to n = 30
# if sd is high cv estimate of accuracy is not that reliable

# examining individual tuples
grid.grid_scores_[0].parameters # dictionary containing parameters used  
grid.grid_scores_[0].cv_validation_scores # validation scores generated using that parameter, can take mean of this to get the average score
grid.grid_scores_[0].mean_validation_score #mean of the above scores


# create a list of the mean scores
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validation Accuracy')

# examine the best model
grid.best_score_ 
grid.best_params_ # dictionary containing parameters used to generate best score
grid.best_estimator_ # actual model object fit with the best parameters


### Searching Multiple parameters Simultaneously ###

# define parameter values that should  be searched
k_range = list(range(1,31))
weight_options = ['uniform', 'distance'] # 2 different options, uniform: all points have equal importance in KNN, weighted: closer points more importance

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors = k_range, weights = weight_options)
param_grid

# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(X, y)

# view the complete results
grid.grid_scores_

# examine best model
grid.best_score_
grid.best_params_


### Using best parameters to make predicts ###

# train your model using all the data and the best known parameters
knn = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')
knn.fit(X,y)

# make a prediction for out of sample data
knn.predict([3,5,4,2])

# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([3,5,4,2])


### Reducing computational expense using RandomizedSearchCV ###

from sklearn.grid_search import RandomizedSearchCV

# specify "parameter distributions" rather than "parameter grid"
param_dist = dict(n_neighbors = k_range, weights = weight_options)

# n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10, random_state = 5)
rand.fit(X,y)
rand.grid_scores_


# examing best model
rand.best_score_
rand.best_params_

# run RandomizedSearchCV 20 times (with n_iter = 10) and record the best score

best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10)
    rand.fit(X,y)
    best_scores.append(round(rand.best_score_,3))
    
best_scores

# switch from Grid Search to RandomizedSearchCV if Grid Search taking too long



### EVALUATING A CLASSIFICATION MODEL ###

import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv(url, header = None, names = col_names)

pima.head()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

X = pima[feature_cols]
y = pima['label']

# split X and y into training and testing set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make class prediction for the training set
y_pred_class = logreg.predict(X_test)

# calculate accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# examine the class distribution of the the testing set (using a Pandas Series method)

y_test.value_counts()


# calculate the % of ones
y_test.mean()
1- y_test.mean() #Null accuracy


# classification accuracy does not tell us anything about the underlying distribution of the testing set


# calculate NULL accuracy in a single line (for binary)
max(y_test.mean(), 1 - y_test.mean())

# NULL accuract for multi class classification
y_test.value_counts()/len(y_test)



## Comparing true and predicted response values

'True:', y_test.values[0:25]
'Pred:', y_pred_class[0:25] # issue handled by confusion matrix

## Confusion matrix

metrics.confusion_matrix(y_test, y_pred_class) # norm: put the true value before the predicted value

#             Predicted
#                0      1                
# Actual 0      TN     FP
#        1      FN     TP

# (FP) = Type 1 error
# (FN) = Type 2 error

confusion = metrics.confusion_matrix(y_test, y_pred_class)

TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

## Classification accuracy

(TP + TN)/float(TP+TN+FP+FN)

metrics.accuracy_score(y_test, y_pred_class) #same as above


## Classification error (Misclassification Rate)

(FP + FN)/float(TP + TN + FP + FN)

1 - metrics.accuracy_score(y_test, y_pred_class) # same as above

## Sensitivity: When actual value is +ve, how often is the prediction correct
# Also TPR or Recall

TP/(TP + FN)

metrics.recall_score(y_test, y_pred_class) #same as above

## Specificity: When actual value is -ve, how often is prediction correct

TN / (TN + FP)

## False Positive Rate (FPR): When actual value is -ve how often is the prediction incorrect

FP / (TN + FP)

## Precision: When a positive value is predicted, how often is the prediction correct

TP / (TP + FP)
metrics.precision_score(y_test, y_pred_class) #same as above

# Can calculate the F1 score and Mathew's correlation coefficient from the confusion matrix


### Adjusting classification threshold ###

# print the first 10 predicted responses
logreg.predict(X_test)[0:10]

# print the first 10 predicted probabilities of class membership
logreg.predict_proba(X_test)[0:10,:] #outputs predicted probabilities of class membership

# print the first 10 predicted probabilities for class 1
logreg.predict_proba(X_test)[0:10, 1]

# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# histogram of predicted probabilities
plt.hist(y_pred_prob, bins = 8)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')

# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob, 0.3)[0]

confusion #previous confusion matrix

metrics.confusion_matrix(y_test, y_pred_class)

# sensitivity has increased

# specificity has decreased


### ROC Curve and Area under the Curve (AUC) ###


# first argument is true values, second the predicted probabilities

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity')
plt.ylabel('True Positive Rate (Sensitivity)')

# Define a function that accepts a threshold and prints sensitivity and specificity

threshold = 1
'Sensitivity:', tpr[thresholds > threshold][-1]

def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
evaluate_threshold(0.5)

## Area Under the Curve (AUC)

## first true values, second predicted values

metrics.roc_auc_score(y_test, y_pred_prob)

# Calculate cross-validate AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv = 10, scoring = 'roc_auc').mean()


## Confusion matrix advantage
# - allows you to calculate a variety of metrics
# - used for multi class problems

## ROC-AUC advantages
# - does not require you to set a classification threshold
# - still useful when there is a high class imbalance