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