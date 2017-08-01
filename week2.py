"""
Created on Tue Jan 10 15:10:28 2017

@author: Mayank Singh
"""

"""***Series data structure***"""

import pandas as pd

animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)

numbers = [1,2,3]
pd.Series(numbers)

animals = ['Tiger', 'Bear', None]

pd.Series(animals)

numbers = [1,2, None]

import numpy as np

np.nan == None

#nan is similar to none but its a numeric value and is treated differently

sports = {'Archery':'Bhutan',
    'Golf': 'Scotland', 
    'Sumo': 'Japan',
    'Taekwondo': 'South     Korea'}

s = pd.Series(sports)
s

#can access objects using index attribute

s.index

#can separate index creation by passing the index explicitly as a list

s2 = pd.Series(['Tiger', 'Bear', 'Moose'], index = ['India', 'America', 'Canada'])

s2

#pandas overrides automatic creation to favor only index values provided
#All keys not in your index will be ignored and Nan or None will be included in their place

sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}

s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])



"""***QUERYING A SERIES***"""
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
          
s = pd.Series(sports)

#iloc and loc are not methods but parameters, so we use [] and not ()
s.iloc[3] #to index with position

s.loc['Golf'] #to index with label

#don't use indexing using just []
#like
s[1]
s['Golf']

#error using above
sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
          
s = pd.Series(sports)

s[0]#error

s.iloc[0] #use this

#An operation on all the items of the series

s = pd.Series([100.00, 120.00, 101.00, 3.00])

total = 0
for item in s:
    total += item
print(total)

#Vectorization

import numpy as np

total = np.sum(s)

#using random to generate a large series

s = pd.Series(np.random.randint(0,1000,10000))

s.head() #displays first 5 rows

len(s)

#cellular magic functions
#to time the code

#on for loop
%%timeit -n 100
summary = 0
for item in s:
    summary+=item
    
%%timeit -n 100
summary = np.sum(s)

#on vectorization
%%timeit -n 100
summary = np.sum(s)


"""broadcasting in numpy, apply an operation to every value in the series, changing the series"""

#increasing every value by 2
s +=2
s.head()

#traditional way
for label, value in s.iteritems():
    s.set_value(label, value+2)
    
s.head()

%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,10000))
for label, value in s.iteritems():
    s.loc[label]= value+2
    
#vectorizing much faster
%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,10000))
s+=2

"""Indices can have mixed type"""
#mixed values and indices are no problem for pandas
s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'

#index values not unique

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries) 
#append to add this data to old one, also append creates a new series made up of the two old series together

cricket_loving_countries

all_countries.loc['Cricket'] #you don't get one value but a series


"""DATAFRAME Data Structure"""
#2D series object

import pandas as pd

#creating a dataframe
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})

purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})

purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()

#extracting data
df.loc['Store 2']

#checking data type
type(df.loc['Store 2'])

#non unique indices(column or row axis)
df.loc['Store 1']

#extracting column
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

#simply put the column name in []
df['Item Purchased']
#all the names and cost values using .loc
df.loc[:,['Name', 'Cost']]


#2 parameters to .loc, finding cost for Store1
df.loc['Store 1', 'Cost']

#chaining operations
#chaining has its problems, use another better method if possible
#chaining returns a copy of the data. Any change made to the data can return errors as the copy will be changed
df.loc['Store 1']['Cost'] #same


#Transpose of the Dataframe
df.T


df.T.loc['Cost'] #using this to extract column
df['Cost'] #much better

"Dropping data"
df.drop('Store 1') #doesn't drop the axis from the df by default but returns a copy of the dataframe with the axis removed

#copying df into copy_df
copy_df = df.copy()

#dropping and assigning back
copy_df = copy_df.drop('Store 1')

#drop has two parameters
#1. inplace = True, data frame will be updated instead of a copy being returned
#2. axes to be dropped, (default)0 -> row, 1 -> column

#deleted without giving a view
del copy_df['Name']

#adding a new column location with default value None
df['Location'] = None

"""DATAFRAME INDEXING & LOADING"""
costs = df['Cost']

#broadcasting to increase cost
costs += 2

#costs in the orginal dataframe will rise as well
df

"""Loading Data"""

#reading a csv file
df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/olympics.csv')

df.head()

#parameters of read_csv
#using index col to set which col has index vals and skip row to start reading it from row2
df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/olympics.csv', index_col=0, skiprows=1)

#setting column name
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True)

#inplace = True updates the df directly        
df.head()


"""***QUERYING A DATAFRAME***"""

#Boolean masking: (Heart of efficient querying in NumPy) Its an array which can be 1D like a series or 2D like a Dataframe, where each value is T or F
#overlaid on top of the data structure, any cell aligned with T value will be admitted to the final result, value with F won't

#what the boolean masking looks like
df['Gold'] > 0

#overlaying that mask on the dataframe
#using where fn -> applies a boolean mask to the df, and returns a df of the same shape

only_gold = df.where(df['Gold'] > 0)

only_gold['Gold'].count() #counting rows meeting criteria

#drop NA values
only_gold = only_gold.dropna()

only_gold.head()

#using boolean mask in the indexing operator

only_gold = df[df['Gold']>0] #same as using where

len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)]) #OR operator

df[(df['Gold.1'] > 0) & (df['Gold'] == 0)] #AND operator


"""***INDEXING DATAFRAMES***"""

#Set index fn, doesn't keep current index

df['country'] = df.index

#setting gold as the index
df = df.set_index('Gold')

#reset index completely, creates a default numbered index

df = df.reset_index()

df.head()


#Census data

df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/census.csv')

df.head()

#unique values
df['SUMLEV'].unique()

df = df[df['SUMLEV'] == 50]

df.head()


columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']

df = df[columns_to_keep]

df.head()

#multilevel indexing (Hierarchical Indices)
df = df.set_index(['STNAME','CTYNAME'])

df.head()

#quering this multi-index data
#outermost column level 0

df.loc['Michigan', 'Washtenaw County']

df.loc[ [('Michigan', 'Washtenaw County'),
         ('Michigan', 'Wayne County')] ] #two counties together
        

         
#Another example

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

#setting multilevel index
df = df.set_index([df.index, 'Name'])

#renaming index
df.index.names = ['Location', 'Name']

#appending new row to the df
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))



"""***MISSING VALUES***"""

#na_values list

#na_filters to turn off white space filtering

#loading file log.csv
df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/log.csv')

df.fillna?
#f fill forward filling, fills value from previous row

#promote time to index
df = df.set_index('time')

#sort by index
df = df.sort_index()

#multilevel indexing to deal with that duplicate indices

df = df.reset_index()

df = df.set_index(['time', 'user'])

#filling missing values with a series, same length as your dataframe

#statistical functions on dataframe generally ignore missing values
