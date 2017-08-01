"""
Created on Wed Jan 18 18:46:40 2017

@author: mayank
"""

"""***READING Tabular data into pandas***"""


import pandas as pd

#by default tab separated, first row header row
pd.read_table('http://bit.ly/chiporders')

user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

#using '\' as separator, no header, giving a list as header names
user = pd.read_table('http://bit.ly/movieusers', sep = '|', header = None, names = user_cols)

user.head()


#skip footer or skip header to skip rows at the top or the bottom


"""***Selecting pandas series Dataframe***"""

ufo = pd.read_csv('http://bit.ly/uforeports')

type(ufo)

ufo.head()

"""bracket notation"""

#case sensitive
ufo['City']

"""Dot notation"""

#everytime a series is added to a dataframe the column name becomes an attribute of the dataframe
ufo.City

#Columns with blank name dont work with dot notation, have to use bracket notation
#also the case if you name a column the same as a built in attribute
ufo['Colors Reported']

'ab' + 'cd'

#use bracket notation when trying to assign values to a new column(series) in a dataframe
ufo['Location'] = ufo.City + ',' + ufo.State #concatenates the values of the two columns

   
"""***Parenthesis in Pandas***"""


movies = pd.read_csv('http://bit.ly/imdbratings')

movies.head()

#will give you descriptive stats of all numeric columns in the dataframe
movies.describe()

#shape of the df, rows X cols
movies.shape

#type of all the columns
movies.dtypes

type(movies)

#movies is a df and as a df it has certain methods and attributes

#the Methods are like head() and describe() - Actions

#the Attributes are without () - Description

#methods have required and optional arguments

#only describes the columns with the type object
movies.describe(include = ['object'])

#shift + tab inside the parenthesis to get the arguments of the method
movies.describe()


"""***Renaming Columns***"""

ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.head()

#to see column names
ufo.columns

#Method 1
ufo.rename(columns = {'Colors Reported': 'Colors_Reported', 'Shape Reported': 'Shape_Reported'}, inplace = True)

#Method 2

ufo_cols = ['city', 'colors reported','shape reported', 'state', 'time']

ufo.columns = ufo_cols #overwrites the column names

ufo.head()

#Method 3
#rename while reading in the file

ufo = pd.read_csv('http://bit.ly/uforeports', names = ufo_cols, header = 0) #zeroth row of the file has existing columns names and you overwrite them


#Method 4 
#to replace a particular character from the names

ufo.column = ufo.columns.str.replace(' ', '_') #replace ' ' with '_' using string method


"""***Removing columns***"""

ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.shape

#using drop method
ufo.drop('Colors Reported', axis = 1, inplace = True)

#axis 0 is row axis
#axis 1 is column axis

#dropping multiple columns at once
#pass list instead of one name
ufo.drop(['City', 'State'], axis = 1, inplace = True)


#removing rows instead of columns
#use labels/indices
ufo.drop([0,1], axis = 0, inplace = True)


"""***Sorting data in DF/Series***"""

import pandas as pd

movies = pd.read_csv('http://bit.ly/imdbratings')

movies.head()

#sorting a series
movies.title.sort_values()

movies['title'].sort_values() #same as above

#sort_values is a series method

#descending order
movies.title.sort_values(ascending = False)

#sort_values doesn't change the underlying data

#sorting the dataframe by a series
movies.sort_values('title') #sorted in alphabetical order of the title

movies.sort_values('duration', ascending = False)

#sorting df by multiple columns
movies.sort_values(['content_rating', 'duration'])



"""***Filter rows by Column value***"""

movies = pd.read_csv('http://bit.ly/imdbratings')

movies.head()

movies.shape

#filter df by duration >200
#Boolean masking

#STEP 1: Creating the boolean masking


#trying it with a for loop to create boolean masking

booleans = []

for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)
        
booleans[0:5] #first 5 objects of booleans

len(booleans)

#STEP 2: Convert booleans list to a pandas series

is_long = pd.Series(booleans)

#Step 3: Pass is_long to the dataframe using bracket notation

movies[is_long]


#using a simple code to get the same result

#Step 1: Create the Boolean Masking
is_long = movies.duration >= 200

is_long.head()

#Step 2: Apply is_long to df

movies[is_long]

#Simplify code further

movies[movies.duration >= 200]

#extracting a column from the filtered data frame

movies[movies.duration >= 200].genre
      
#using .loc for the same
#.loc selects rows and columns
movies.loc[movies.duration >= 200, 'genre']


"""***Multiple filter criteria to DF***"""

import pandas as pd

movies = pd.read_csv('http://bit.ly/imdbratings')

movies[movies.duration >= 200]

#200+ mins movies of the genre 'Drama'

#and
movies[(movies.duration >= 200) & (movies.genre == 'Drama')]

#or
movies[(movies.duration >= 200) | (movies.genre == 'Drama')]

#or in the column
#movies either Action or Drama or Action using .isin()

movies[movies.genre.isin(['Crime', 'Drama', 'Action'])]

"""***Q&A***"""

#reading csv ignoring columns

ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.columns

#only reading in City and State columns
#reference by name
ufo = pd.read_csv('http://bit.ly/uforeports', usecols = ['City', 'State'])

ufo.columns

#reference by position
ufo = pd.read_csv('http://bit.ly/uforeports', usecols = [0,4])

#reading csv faster
#only need a first 3 rows

ufo = pd.read_csv('http://bit.ly/uforeports', nrows = 3)


#df and series are iterable

#iterating thorough the series
for c in ufo.City:
    print(c)
    
    
#iterating through the dataframe
for index, row in ufo.iterrows():
    print(index, row.City, row.State)
    
    
    
#dropping every non numberic column from a df

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.dtypes

import numpy as np

drinks.select_dtypes(include = [np.number]).dtypes
                    

#using [] with include
#check the describe
drinks.describe(include = 'all')

drinks.describe(include = ['object', 'float64'])


"""***Axis parameter in pandas***"""

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.head()

#dropping column continent
drinks.drop('continent', axis = 1).head()

#dropping row 2
drinks.drop(2, axis = 0).head()


#column-wise mean
drinks.mean(axis = 0) #default anyways

#row-wise mean
drinks.mean(axis = 1)


#can also use
drinks.mean(axis = 'index') #instead of 0

drinks.mean(axis = 'columns') #instead of 1

           
"""***String methods in pandas***"""

#making it upper case
'hello'.upper()

orders = pd.read_table('http://bit.ly/chiporders')

orders.head()

#to use string methods on a pandas series add .str

orders.item_name.str.upper()

orders.item_name.str.contains('Chicken')

#df filtered with Chicken in the item name
orders[orders.item_name.str.contains('Chicken')]

#chaining together string methods

#removing '[]' from the choice_description column
orders.choice_description.str.replace('[', '').str.replace(']', '')


#many pandas string methods will accept Regex (Regular expressions)

orders.choice_description.str.replace('[\[\]]', '')


"""***Changing data type of pandas Series***"""

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.head()

drinks.dtypes

#convert beer_servings column to float64 instead of int64

drinks['beer_servings'] = drinks.beer_servings.astype(float)

drinks.dtypes

#defining type of each column while reading the csv
drinks = pd.read_csv('http://bit.ly/drinksbycountry', dtype = {'beer_servings':float})

drinks.dtypes

orders = pd.read_table('http://bit.ly/chiporders')

orders.head()

orders.dtypes

orders['item_price'] = orders.item_price.str.replace('$', '')

orders.dtypes

orders['item_price'] = orders.item_price.astype(float)


#getting the boolean mask as 0s and 1s instead of true and false
orders.item_name.str.contains('Chicken').astype(int).head()


"""***Group By in pandas***"""

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

#avg beer_servings across all countries
drinks.beer_servings.mean()

#avg beer_servings per continent
drinks.groupby('continent').beer_servings.mean()


#filter data for africa and then mean
drinks[drinks.continent == 'Africa'].beer_servings.mean()

#other aggregation fns
drinks.groupby('continent').beer_servings.max()

drinks.groupby('continent').beer_servings.min()

#.agg() can be used to pass multiple aggregation fns
drinks.groupby('continent').beer_servings.agg(['count','mean', 'min', 'max'])

#if column is not specified the measure will be calculated along all numeric columns
drinks.groupby('continent').mean()

#plotting a chart using the above data

%matplotlib inline
drinks.groupby('continent').mean().plot(kind = 'bar')

"""***Exploring Pandas Series***"""

movies = pd.read_csv('http://bit.ly/imdbratings')

movies.head()
movies.dtypes

#.describe()
movies.genre.describe()

#counts the no. of times a value appears in a series
movies.genre.value_counts()

#getting the same in percentages instead of counts
movies.genre.value_counts(normalize = True)

#output of series/df methods are usually series/dfs and can be chained together with other methods already known

#shows all the unique values in the series
movies.genre.unique()

#no. of unique values in the series
movies.genre.nunique()


#cross tabulation
#genre as row headers, content_rating as column headers
pd.crosstab(movies.genre, movies.content_rating)

#different output with .describe() for numeric columns
movies.duration.describe()

#standard deviation
movies.duration.std()

#can be used with numeric columns as well
movies.duration.value_counts()

%matplotlib inline
movies.duration.plot(kind = 'hist')

movies.genre.value_counts().plot(kind = 'bar')



"""***Missing values***"""

ufo = pd.read_csv('http://bit.ly/uforeports')

#last 5 values
ufo.tail()

#shows a false if something is not null(missing)
ufo.isnull().tail()

#opposite of the above
ufo.notnull().tail()


#no. of missing values in each column
ufo.isnull().sum()

ufo.isnull().sum(axis = 0) #same as above

#using boolean masking to show only Null city values from the ufo df
ufo[ufo.City.isnull()]


#drop missing values

#drop a row if any of its values are missing
ufo.dropna(how = 'any').shape

#drop NAs from only if all the values are missing, inplace present here too
ufo.dropna(how = 'all').shape

#drop rows if a particular column contains NAs
ufo.dropna(subset=['City', 'Shape Reported'], how = 'any').shape #drop a row if City or Shape Reported are NA

ufo.dropna(subset=['City', 'Shape Reported'], how = 'all').shape #drop a row if City and Shape Reported are NA         
          

ufo['Shape Reported'].value_counts() #by default missing values are excluded

#unless
ufo['Shape Reported'].value_counts(dropna = False)

#fill missing 'Shape Reported' with Various

ufo['Shape Reported'].fillna(value = 'VARIOUS', inplace = True)

ufo['Shape Reported'].value_counts(dropna = False)


"""***Indices in Pandas***"""

import pandas as pd

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.head()

drinks.index #also known as row labels

drinks.columns

pd.read_table('http://bit.ly/movieusers', header = None, sep = '|').head()

#indices used for selection, identification and alignmenft

drinks[drinks.continent == 'South America']

drinks.loc[23, 'beer_servings']

#changing index
drinks.set_index('country', inplace = True)

drinks.head()

drinks.index

drinks.shape

#now using Brazil as index instead of a number
drinks.loc['Brazil', 'beer_servings']


#name of the index is also displayed, no need for an index name but its helpful


drinks.index.name = None #remove the name of the index

drinks.head()

#setting default index and move index to a columns

#give name back
drinks.index.name = 'country'

#reset index
drinks.reset_index(inplace = True)

drinks.head()

drinks.describe() #is a df and has an index

drinks.describe().index

drinks.describe().loc['25%', 'beer_servings']

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.set_index('country', inplace = True)

#country is the index
drinks.continent.head()

#also a series
drinks.continent.value_counts()

drinks.continent.value_counts().index

#we can use this index to select values from these series as well

#use index Africa and display that value
drinks.continent.value_counts()['Africa']

#sort by value_counts in ascending order by default
drinks.continent.value_counts().sort_values()

#sort by index in ascending order
drinks.continent.value_counts().sort_index()

"""Alignment using index"""

people = pd.Series([3000000, 8500], index = ['Albania', 'Andorra'], name = 'population')

#calculate the total beer servings of each country

#people X beer_servings in drinks df

#alignment puts data together even if its not the same length
drinks.beer_servings * people #for countries not in people df resulting multiplication yields NaN

#adding this people series to the drinks df

#pd.concat can be used to concatenate rows and columns using the axis parameter, axis = 1 concat using columns, axis = 0 concat rows
pd.concat([drinks, people], axis = 1).head()


"""***.loc[] , .iloc[] and .ix[]***"""

#different df methods for selecting rows and columns

ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.head(3)


""" .loc """
#used to select info by labels, rows - index, columns - column name
ufo.loc[0,:] #row 0 and all columns

ufo.loc[[0,1,2],:] #rows 0, 1,2 and all columns

#same as above
ufo.loc[0:2,:]

#the above notations are inclusive on both sides, unlike range

ufo.loc[0:2] #same as the code above but harder to read
#explicit is better than implicit

ufo.loc[:,'City'] #all rows for column cities

ufo.loc[:, ['City', 'State']] #all rows for col City and State

ufo.loc[:, 'City':'State'] #all rows and cols from City and State

ufo.loc[0:2, 'City':'State'] #combining the two

#same as above
ufo.head(3).drop('Time', axis = 1) #lots of ways to do the same things


#all rows with city = Oakland
ufo[ufo.City == 'Oakland']


#another way to do it
ufo.loc[ufo.City == 'Oakland',:]

ufo.loc[ufo.City == 'Oakland','State'] #one internal operation

ufo[ufo.City == 'Oakland'].State #same but its chained indexing and might cause problems in certain cases. Two operations


""".iloc"""
#filtering rows and selecting columns by integer positions

ufo.iloc[:, [0,3]]

ufo.iloc[:, 0:4] #shows columns in position 0, 1, 2 and 3. Exclusive of the second number in the range, inclusive of the first number

ufo.iloc[0:3, :] #rows 0,1 and 2 and all columns


#some shortcuts
ufo[['City', 'State']] #not good practice to select two columns

ufo.loc[:, ['City', 'State']] #better practice

ufo[0:2] #not good practice to select rows

ufo.iloc[0:2,:] #much better

""" .ix """

#allows us to mix labels and integers

drinks = pd.read_csv('http://bit.ly/drinksbycountry', index_col = 'country')

drinks.ix['Albania', 0] #first one is label(row index of cell) and integer(column of cell)

drinks.ix[1, 'beer_servings'] #1(position of row), beer_servings(label)

drinks.ix['Albania':'Andorra', 0:2] #if you have a string index(in this case for the column) and you have numbers in the code then the no.s are treated as positions, so inclusive of position 0 and exclusive of 2 (like .iloc)

ufo.ix[0:2, 0:2] #but here the index of the rows is an integers if it was country then only 2 rows would have been returned, so inclusive on both sides


"""***Inplace parameter in pandas***"""

import pandas as pd

ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.shape

ufo.head()

ufo.drop('City', axis = 1).head() #not actually dropped from the original df

ufo.head()

ufo.drop('City', axis = 1, inplace = True)
#if nothing prints out, then its a tipoff in python that something has happened inplace

ufo.head()

ufo.dropna(how = 'any')

ufo.rename()

ufo.sort_values()

ufo = ufo.set_index('Time') #same as ufo.set_index('Time', inplace = True)

#assignment has the tendency to create two copies momentarily, if its a big df then there might be problems
#no guarantee that inplace is more efficient though

#inplace = False good way to explore without affecting the underlying df
ufo.fillna(method = 'bfill').tail()

ufo.fillna(method = 'ffill').tail()



"""***Making Dataframes Faster and Smaller***"""

import pandas as pd

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.head()

drinks.info() #more info about the df columns

#you can make a pandas series of python lists or python dictionaries

#object usually means a string is being stored

drinks.info(memory_usage = 'deep') #for understaning what the object size is for real with the df and not just for the references to those objects

#space each column takes
drinks.memory_usage()

#actual size not just the reference to objects
drinks.memory_usage(deep = True)

#can carry out series operations
drinks.memory_usage(deep = True).sum()

"""More space efficient with object columns"""

#IDEA1:
#if you can store your object columns as integers it will be more space efficient

sorted(drinks.continent.unique())

#instead of storing strings we can store integers to mean those strings

#0 - Africa, 1 - Asia, 2 - Europe...

drinks.continent.head()

#will still have to store a lookup table, will have to store the strings only once

#can be done with Category type

drinks['continent'] = drinks.continent.astype('category')

drinks.dtypes #continent is category type

drinks.continent.cat.codes.head() #similar to .str we saw earlier for strings, this is used for category variables

#continent series is represented as integers now

drinks.memory_usage(deep = True)

#repeat for country

drinks['country'] = drinks.country.astype('category')

drinks.country.cat.codes.head()

drinks.country.head()

#country gets larger simply because we created 193 categories, an additional lookup table with all those categories
drinks.memory_usage(deep = True)

#this is useful when you are using repeated values with few uniques

#also speeds of computations

df = pd.DataFrame({'ID':[100, 101, 102, 103], 'quality': ['good', 'very good', 'good', 'excellent']})

df

df.sort_values('quality')

#telling pandas there is a logical ordering
#use catgory data type and define ordered categories

df['quality'] = df.quality.astype('category', categories=['good', 'very good', 'excellent'], ordered = True)

df.sort_values('quality')

#can do this as well with the ordered categorical type
df.loc[df.quality > 'good']



"""***Using Pandas with scikit-learn for machine learning***"""

train = pd.read_csv('http://bit.ly/kaggletrain')

#for the test set predict survival based upon other characteristics of the passengers


#Step1: Create Feature Matrix x
#these are the features, the columns that our model is going to learn from

feature_cols = ['Pclass', 'Parch']

X = train.loc[:, feature_cols]

X.shape

#Step2: Create Response/Target Vector
#what you are trying to predict

y = train.Survived

y.shape

#scikit-learn will understand the x and y objects as long as they are all numeric, and the right shape

#code to create Classification model
#fitted the ML model to the train data
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)

#Read the test dat to make predictions on
test = pd.read_csv('http://bit.ly/kaggletest')

test.head() #survived column missing

X_new = test.loc[:, feature_cols]

X_new.shape

#what are the predicted classes for the test data
new_pred_class = logreg.predict(X_new)

test.PassengerId

new_pred_class

#create a Dataframe and write it to a csv
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived': new_pred_class}).set_index('PassengerId').to_csv('sub.csv')

#dictionaries are unordered so its not possible to gurantee that passengerId comes first

#to ensure its first is set it as index


"""Saving python object to a disk"""

train.to_pickle('train.pkl') #can load it directly from a flash drive without going through all the steps required to create the dataframe

#reading the pickle file
pd.read_pickle('train.pkl')


"""***Extra content***"""

#choose the pandas doc with the latest version

#pandas.(...) top level function

#pandas.DataFrame.(...) dataframe method to run it you will need to use df.drop

#API Reference - VERY IMPORTANT
#list of all the functions in pandas
#list string methods very imp

"""Difference b/w pd.isnull(obj) & df.isnull()"""

ufo = pd.read_csv('http://bit.ly/uforeports')

pd.isnull(ufo).head()

ufo.isnull().head()

"""Random sampling"""

ufo.sample(n = 3, random_state = 42) #you'll 3 random rows
#with random_state reproducibility can be achieved (like seed in R programming)

ufo.sample(frac = 0.75, random_state = 99) #75% of the rows are ouputted

          
"""ML Train-Test Split"""
#certain percentage of rows in train set and certain in test set with both being exclusive

train = ufo.sample(frac = 0.75, random_state = 99)

#all the other rows (25%) for the testing set

test = ufo.loc[~ufo.index.isin(train.index), :]
# ~ inverts the series, Trues become Falses and vice versa
test.head()

test.shape

train.shape


"""***Dummy variables in Pandas***"""

train = pd.read_csv('http://bit.ly/kaggletrain')

train.head()

#creating dummy variable for the sex column shown in the df

train.dtypes

train['Sex_male'] = train.Sex.map({'female':0, 'male': 1})

train.head()

#another way that is more flexible

pd.get_dummies(train.Sex)

pd.get_dummies(train.Sex, prefix = 'Sex').iloc[:,1:] #creates one column for each possible value, drops first column, prefixes sex and adds '_' something else can be specified

              
train.Embarked.value_counts()
pd.get_dummies(train.Embarked, prefix = 'Embarked')

#if you have n possible values, you need n-1 dummy variables to capture the info about that feature 


embarked_dummies = pd.get_dummies(train.Embarked, prefix = 'Embarked').iloc[:, 1:] #dropped C column(first column)
#Q is the base line
# 0-0 C
#0-1 S
#1-0 Q

#concatenated to the dataframe
train = pd.concat([train, embarked_dummies], axis = 1)

train.head()


#rerun train definition before the below code

#dummies for dataframe and not series
pd.get_dummies(train, columns = ['Sex', 'Embarked'], drop_first = True)

#drop = True drops the first dummy variable column automatically



"""***Date and Time***"""

import pandas as pd

ufo = pd.read_csv('http://bit.ly/uforeports')

ufo.head()

ufo.dtypes

#slicing to get the hour, not efficient though
ufo.Time.str.slice(-5,-3)

#better to convert time column to pandas date-time format

ufo['Time'] = pd.to_datetime(ufo.Time)

ufo.head()

ufo.dtypes

#to get hour
ufo.Time.dt.hour

ufo.Time.dt.weekday_name

ufo.Time.dt.dayofyear

#API Reference search ".dt." and check out other properties

ts = pd.to_datetime('1/1/1999') #creates a timestamp

#can use them for comparison

ufo.loc[ufo.Time >= ts, :].head()

#mathematical operations with date-time format

ufo.Time.max() #latest time stamp

(ufo.Time.max() - ufo.Time.min()).days #outputs a time delta


#plot of no. of ufo reports by year

%matplotlib inline
ufo['Year'] = ufo.Time.dt.year
   
ufo.head()

#plot of ufo sightings by year
ufo.Year.value_counts().sort_index().plot()



"""***Removing Duplicates***"""

import pandas as pd

# read a dataset of movie reviewers (modifying the default parameter values for read_table)
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

users = pd.read_table('http://bit.ly/movieusers', sep='|', header=None, names=user_cols, index_col = 'user_id')

users.head()

users.shape

users.zip_code.duplicated() #series of Trues and Falses, True if an entry above it that is identical
#it won't tell which one is it duplicate of

users.zip_code.duplicated().sum()

users.duplicated() #true if an entire row is same as a row above it

users.duplicated().sum()

users.loc[users.duplicated(), :] #to see the duplicated rows

users.loc[users.duplicated(keep = 'first'), :] #default

#logic for first - mark duplicates as true, except for the first occurance
#last - later duplicates are kept and above ones are outputted as true

users.loc[users.duplicated(keep = 'last'), :]

#outputs all the duplicates as True
users.loc[users.duplicated(keep = False), :]

users.drop_duplicates(keep = 'first').shape
                
#age + zipcode is a unique combination and their duplicates should be removed
users.duplicated(subset = ['age', 'zip_code']).sum()
#only age and zip_code as the relevant columns for duplicate finding 

users.drop_duplicates(subset = ['age', 'zip_code']).shape
                      

"""***SettingWithCopyWarning***"""

#Scenario1:

movies = pd.read_csv('http://bit.ly/imdbratings')

movies.head()

#missing values for content_rating
movies[movies.content_rating.isnull()]

movies.content_rating.value_counts()

#not rated should be replaced with missing values

movies[movies.content_rating == 'NOT RATED'].content_rating
      
import numpy as np

#SettingWithCopy Warning
movies[movies.content_rating == 'NOT RATED'].content_rating = np.nan
      
movies.content_rating.isnull().sum() #didn't work

movies.loc[movies.content_rating == 'NOT RATED', 'content_rating'] = np.nan
#.loc specified the row and the column

movies.content_rating.isnull().sum() #worked

#WHY: first line of code is two operations pandas can't guarantee if the get item produced a view or a copy of the data

#.loc makes it a single operation and then no error

#if you are trying to select rows and columns in the same row then use .loc


#SecondExample: SettingWithCopy Warning

top_movies = movies.loc[movies.star_rating >= 9, :]

#same setting with copy warning even with .loc
top_movies.loc[0, 'duration'] = 150
              
#it does actually modify
#pandas isn't sure if top_movies is a view or a copy of movies, so its warning if you are trying to modify top_movies or both top_movies and movies

#problem here
#anytime you are creating a df copy, explicitly use .copy() method, pandas is sure its a copy
top_movies = movies.loc[movies.star_rating >= 9, :].copy()

#no warning
top_movies.loc[0, 'duration'] = 150
              

"""***Display options in pandas***"""

import pandas as pd

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks #first 30 rows then last 30 rows

#want to show all rows

#check pandas.get_option to get display options

pd.get_option('display.max_rows') #to see the default

pd.set_option('display.max_rows', None) #None shows all rows

drinks

#reset the display options

pd.reset_option('display.max_rows')

drinks

#max no. of columns displayed
pd.get_option('display.max_columns')

pd.set_option('display.max_columns', None)

pd.reset_option('display.max_columns')

train = pd.read_csv('http://bit.ly/kaggletrain')

train.head()

#max limit to the number of characters displayed in a cell, Florence Briggs Th...

pd.get_option('display.max_colwidth')

#can't use None in this case
pd.set_option('display.max_colwidth', 1000)

train.head()

#changing the number of decimal points

pd.get_option('display.precision')

pd.set_option('display.precision',2) #limited to two decimal places
#doesn't affect the data, just the display

train.head()

drinks.head()

drinks['x'] = drinks.wine_servings * 1000

drinks['y'] = drinks.total_litres_of_pure_alcohol * 1000
      
drinks.head()

#adding the comma(,) like 3,000 and 1,000,000 etc

pd.set_option('display.float_format', '{:,}'.format) #passing a python format string, meaning use , as 1000 seperator
#affects only float format

drinks.head()

#reading up on the pandas options

pd.describe_option() #all options displayed

pd.describe_option('rows') #only options with rows in the names

#Resetting all option

pd.reset_option('all')


"""***Creating a Pandas DataFrame***"""

import pandas as pd

"""Creating DF from a dictionary"""

pd.DataFrame({'id':[100, 101, 102], 'color': ['red', 'blue','red']})
#columns not in the same order as written, cause dictionary is an unordered structure

#can order it using the columns argument and specifying index
df = pd.DataFrame({'id':[100, 101, 102], 'color': ['red', 'blue','red']}, columns = ['id', 'color'], index = ['a', 'b', 'c'])


"""List of Lists is passed"""
#each inner list gets treated as a row
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns = ['id', 'color'])

"""Converting a Numpy array to a DataFrame"""

import numpy as np

#check random number fns on "https://docs.scipy.org/doc/numpy/reference/routines.random.html"

arr = np.random.rand(4,2) #4X2 numpy array rand no.s b/w 0 & 1

arr

#converting numpy array to DataFrame
pd.DataFrame(arr, columns = ['one', 'two'])



#DF of 10 rows and 2 cols, Student ID and Test scores

pd.DataFrame({'Student': np.arange(100, 110, 1), 'test': np.random.randint(60, 101, 10)})

#both of them are exclusive of the second number in the argument, randint, frm 60 to 101 and 10 digits, arange 100 - 109, steps of 1



#chain it together with set_index if you have one of the columns as index
pd.DataFrame({'Student': np.arange(100, 110, 1), 'test': np.random.randint(60, 101, 10)}).set_index('Student')


"""Creating a series and attaching it to a DF"""

s = pd.Series(['round', 'square'], index = ['c', 'b'], name = 'shape')

s

df

#combining using concatenate
#concat side by side therefore axis = 1
pd.concat([df, s], axis = 1)

#name of the series becomes the column name in the df
#aligned using the index
#NaN added in the missing cell for index a in shape


"""***Applying functions to Pandas Series***"""

import pandas as pd

train = pd.read_csv('http://bit.ly/kaggletrain')

train.head()


"""map"""
# a series method

#creating a dummy variable for sex in the df

#map allows you to map an existing values of a series to a different set of values

#female to 0, male to 1
train['Sex_num'] = train.Sex.map({'female':0, 'male':1})

train.loc[0:4, ['Sex', 'Sex_num']]


"""apply"""
# a series and a df method

#it applies a fn to each element in a series

#calc the length of each string in col 'Name'

train['Name_length'] = train.Name.apply(len)
#just pass the name of the fn without the ()

train.loc[0:4, ['Name', 'Name_length']]

import numpy as np

#rounding decimal numbers using numpy ceiling fn
    
train['fare_ceil'] = train.Fare.apply(np.ceil)

train.loc[0:4, ['Fare', 'fare_ceil']]


#extract last name of each person into its own column

#split with comma
train.Name.str.split(',').head()

#extracting first element of each series, using a fn

def get_element(my_list, position):
    return my_list[position]

train.Name.str.split(',').apply(get_element, position = 0).head()
#apply fn get_element and pass the keyword argument position as 0

#with the lambda fn
train.Name.str.split(',').apply(lambda x: x[0]).head()


#Apply as a df method

drinks = pd.read_csv('http://bit.ly/drinksbycountry')

drinks.head()

#applies a fn along either axis of a df

drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis = 0)
#apply max fn for each column

drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis = 1)
#max value in each row

#which column is the maximum instead of whats its value, use np.argmax

drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis = 1)


"""Applymap"""
#its a df method
#applies a fn to every element of the dataframe

drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float)
#every element in the df to a floating point