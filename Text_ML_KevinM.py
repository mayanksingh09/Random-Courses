# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 04:57:45 2017

@author: Mayank
"""

import pandas as pd

simple_train = ['hey there delilah', 'Let me hold you', 'Reply back', 'Talk to you later']

# Import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()

# Learn the 'vocabulary' of training data (in place)

vect.fit(simple_train) 

#Five things done:
# 1. token pattern drops one letters
# 2. lowercase
# 3. removes punctuations
# 4. alphabetical order
# 5. no duplicates


#examine fitted vocabulary
vect.get_feature_names()

# transform training data into a 'document-term matrix'

simple_train_dtm = vect.transform(simple_train)


# convert sparse matrix to a dense matrix

simple_train_dtm.toarray()

# examine the vocabulary and document-term matrix together

pd.DataFrame(simple_train_dtm.toarray(), columns = vect.get_feature_names())

# type of the document-term matrix
type(simple_train_dtm)

# examine the sparse matrix contents
print(simple_train_dtm) ## Left: coordinates of non zero values, Right: non zero values


# example for model testing
simple_test = ["please don't call me"]

# in order to make a prediction the new observation must have the same features as the training observations, both in number and in meaning

# transform testing data into a document-term matrix (using existing vocabulary)

simple_test_dtm = vect.transform(simple_test) # drops unknown  words

simple_test_dtm.toarray()


# examine the vocabulary and document-term matrix together

pd.DataFrame(simple_test_dtm.toarray(), columns = vect.get_feature_names())

# we are okay with dropping the new words from the testing set, because we don't know the classification of the new word anyways

## SUMMARY

# - vect.fit(train) : learns the vocabulary of the training data
# - vect.transform(train) : uses the fitted vocabulary to buid a document-term matrix from the training data
# - vect.transform(test) uses the fitted vocabulary to build a document term matrix from the testing data (and ignores new words/tokens)

## can read a text based file using pandas

url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'

sms = pd.read_table(url, header = None, names = ['label', 'message'])

# examine shape
sms.shape

# examine first 10 rows
sms.head(10)

# examine class distribution
sms.label.value_counts()

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

# check the conversion
sms.head(10)

# defining X and y

X = sms['message'] # for use with count vectorizer pull out pandas series (1D) : because its going to transformed to a 2D object by count vectorizer
y = sms['label_num']

print(X.shape)
print(y.shape)

# split X and y into training and testing sets

from sklearn.cross_validation import train_test_split


## carry out train test split before running the count vectorizer, otherwise the document matrix will contain all the words from the training and testing set not correctly representing the real world situations

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


### VECTORIZING DATASET ###

# instantiate the vectorizer
vect = CountVectorizer()

# Learn training data vocabulary, then use it to create 
vect.fit(X_train)
X_train_dtm = vect.fit_transform(X_train)

# equivalently: combine fit and transform into one

X_train_dtm = vect.fit_transform(X_train)

# examine the document-term matrix

X_train_dtm #4179: no. of rows, 7456: tokens/words in the data

# transform testing data (using fitted ovabulary) into a document-term matrix

X_test_dtm = vect.transform(X_test) # only do a transform on the testing set
X_test_dtm



### Building and evaluating a model ###

## Using Multinomial Naive Bayes

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# Train the model using X_train_dtm (also time it)
%time nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class) # checking accuracy

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

#print message text for the false positives (ham incorrectly classifier)
X_test.loc[y_pred_class > y_test]

# print message text for the false negatives (spam incorrectly classifier)
X_test.loc[y_pred_class < y_test]

# calculate predicted probabilities for X_test_dtm
y_pred_prob = nb.predict_proba(X_test_dtm)[:,1]
# nb.predict_proba(): Predicts probability its class 0 or 1

# Naive bayes produces extreme predicted probability values

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)


### Comparing Models ###

## comparing NB of Logistic Regression

# Import and instantiate a Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# train the model using X_train_dtm
%time logreg.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)

# calculate predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)
y_pred_prob


# calculate predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
# good model to use if you care about predicted probabilities or if your evaluation metric is logloss


# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

## can use any classification model for text classification

### Examining a model for further insight ### 

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# examine first 50 tokens
print(X_train_tokens[0:50]) # word characters

# examine last 50 tokens
print(X_train_tokens[-50:])

# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_ # _ for attributes that exist post model fitting, token 00 (first token) occurs 0 times in ham and 5 times in spam

# rows represent classes, columns represent tokens
nb.feature_count_.shape

# for each token nb calculates conditional probability of that token given each class (0 or 1)

# to make a prediction nb calculates conditinal probability of a class given the tokens in that message

# number of times each token appears (not count of number of messages in which it appears) across all HAM messages
ham_token_count = nb.feature_count_[0,:]
ham_token_count

# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1,:]

# create a DF of tokens with their separate ham and spam
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam': spam_token_count})

tokens.head() # df with ham and spam count

# examine 5 random DataFrame rows
tokens.sample(5, random_state = 6)

# for class imbalance deal with it depending upon the type of model

# Naive bayes counts the number of observations in each class

nb.class_count_

## To normalize the data

# add 1 to ham and spam columns to avoid dividing by 0
tokens['ham'] = tokens['ham'] + 1
tokens['spam'] = tokens['spam'] + 1
tokens.sample(5, random_state = 6)

# convert the ham and spam counts into frequency

tokens['ham'] = tokens.ham/nb.class_count_[0]
tokens['spam'] = tokens.spam/nb.class_count_[1]

tokens.sample(5, random_state = 6)

# calculate the ratio of spam to ham for each token
tokens['spam_ratio'] = tokens.spam/tokens.ham # can be used to compare hammy v/s spammy words, but dont use the spam_ratio metric as is to interpret spamminess or hamminess

# examine the DataFrame sorted by spam_ratio

tokens.sort_values('spam_ratio', ascending = False)

# Look up the spam_ratio for a given token
tokens.loc[tokens['token'] == 'dating', 'spam_ratio']

# Practice Problem

# check out the jupyter notebook on: 
# Link: https://github.com/mayanksingh09/pycon-2016-tutorial


### TUNING STRATEGIES ###

# default parameters for CountVectorizer
vect

# remove English stop words
vect = CountVectorizer(stop_words = 'english') # can also pass custom list of stop words

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range = (1,2)) # ngrams used for word pairs
# danger of including two grams: no. of features grow really quickly
# adds noise also, check and see for their value

# ignore terms that appear in more than 50% of the rows
vect = CountVectorizer(max_df = 0.5) #removes words that are too common

# ignore rare terms, keep a term that appears in atleast 2 documents in a corpus
vect = CountVectorizer(min_df = 2)