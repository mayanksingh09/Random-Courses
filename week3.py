"""
Created on Fri Jan 13 16:35:42 2017

@author: mayank
"""

"""***Merging Dataframes***"""

import pandas as pd

df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])

#adding a new column date to the df
#same length as the df

df['Date'] = ['December 1', 'January 1', 'mid-May']

#another column, delivery flag
#scalar value, so it can be a single value(repeated across records)
df['Delivered'] = True

#when we have only few items to add
#have to supply None values

df['Feedback'] = ['Positive', None, 'Negative']

#if indices are unique then we can assign new column identifier to the series 

adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'}) #pandas assigns NaN values to the missing index


"""*Joining larger dataframes together*"""

#staff df
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'}, {'Name': 'Sally', 'Role': 'Course liason'}, {'Name': 'James', 'Role': 'Grader'}])

#name as index
staff_df = staff_df.set_index('Name')

#student df
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])

#name as index
student_df = student_df.set_index('Name')

#both dfs indexed along the values we want to merge them on

"""Union"""

pd.merge(staff_df, student_df, how = "outer", left_index=True, right_index = True)

#Mile Role is NaN, Kelly School is NaN

"""Intersection"""
pd.merge(staff_df, student_df, how = "inner", left_index = True, right_index = True)

"""Left Join"""
pd.merge(staff_df, student_df, how = "left", left_index = True, right_index= True)

pd.merge(staff_df, student_df, how = "right", left_index = True, right_index = True)

"""Joining using columns instead of indices"""

staff_df = staff_df.reset_index()

student_df = student_df.reset_index()

pd.merge(staff_df, student_df, how = "left", left_on = "Name", right_on = "Name")


"""Conflicts in dataframe"""


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])

student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])

pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')

#_x left dataframe info _y right df info, their names can be controlled with additional parameters


"""Multi-indexing and Multiple Columns"""

#first name same but last name might not be

staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])

student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])

#using the list of multiple columns to join keys, on left on and right on parameters

pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])



"""***PANDAS IDIOMS***"""

#idiomatic solution is one which has high performance and high readability

#pandorable 

"""Method chaining"""
import pandas as pd
df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/census.csv')

#chain indexing is a bad practice
#df.loc["Washtenaw"]["Total Population

#Method chaining
(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME', 'CTYNAME'])
    .rename(columns={'ESTIMATEBASE2010': 'Estimates Base 2010'}))

#without method chaining
df = df[df['SUMLEV']==50]

df.set_index(['STNAME','CTYNAME'], inplace=True)

df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})

#drop 0 from the rows and rename a column
(df.drop(df[df['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))

"""MAP Fn in Python"""
#map: pass the fn you want called and the iterable like a list

#applymap: fn that should operate on each cell of a dataframe

#apply: map across all the rows in a dataframe

import numpy as np

"""APPLY"""
#takes a row of data, finds min and max of the data and creates a new row of data
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})
    
#axis is the axis of the index to use, in this case its the columns, so axis = 1
df.apply(min_max, axis=1)

#another solution for the same
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row
    
df.apply(min_max, axis=1)

#Using LAMBDA

rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']

df.apply(lambda x: np.max(x[rows]), axis=1)

"""***GROUP BY***"""
import pandas as pd
import numpy as np

df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/census.csv')
df = df[df['SUMLEV']==50]

#timing the code - for loop

%%timeit -n 10
for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME'] == state).dropna()['CENSUS2010POP'])
    print('Countries in state' + state + 'have an average population of ' + str(avg))
    

#same using GROUPBY

%%timeit -n 10
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Countries in state ' + group + ' have an average population of ' + str(avg))
    

#using a fn along with group by to segment data

#set the index as the column with which you want to group by

df = df.set_index('STNAME')

#1st letter M then 0, and so on
def fun(item):
    if item[0] < 'M':
        return 0
    if item[0] < 'Q':
        return 1
    return 2
    
for group, frame in df.groupby(fun):
    print ('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')
    
#common workflow with gropu by: 1. split your data 2. apply a fn 3. combine the results

#Split-Applying-Combine methods

df.groupby('STNAME').agg({'CENSUS2010POP': np.average})

df.groupby('STNAME').agg('sum') #aggregates all the summable rows


#Applying multiple fns to same iterable groups

print(type(df.groupby(level=0)['POPESTIMATE2010', 'POPESTIMATE2011']))

print(type(df.groupby(level=0)['POPESTIMATE2010']))

#applying two fns to a series
df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average, 'sum': np.sum})

#applying two fns to a df
df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010', 'POPESTIMATE2011'].agg({'avg': np.average, 'sum': np.sum})


#using groupby with apply & lambda
df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity')

#using groupby with apply

 def totalweight(df, w, q):
        return sum(df[w] * df[q])
       
df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity')


"""***SCALES***"""

#Ratio scale
#Interval scale
#Ordinal scale
#Nominal scale

import pandas as pd

df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'], index = ['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])

df.rename(columns={0: 'Grades'}, inplace = True)

#converting to categorical type
df['Grades'].astype('category').head()

grades = df['Grades'].astype('category', categories = ['D', 'D+', 'C-', 'C', 'C+', 'B-','B', 'B+', 'A-', 'A', 'A+'], ordered = True)

grades.head()

#ordinal data has ordering, so it will help with boolean masking
#can use min max on this data
grades > 'C'

df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/census.csv')

import numpy as np
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})

#reducing a value on the interval or ratio scale(Number grade) to a categorical scale(Letter grade)


#cut gives you interval data with spacing b/w each category has equal size
pd.cut(df['avg'],10) #argument - column or array like series, no. of bins to be used - all bins at equal spacing

#categories w/ no. of items in each bin as same instead of spacing being same (check how)


"""***PIVOT TABLE***"""

df = pd.read_csv('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/cars.csv')

#comparing battery capacity by year and vehicle

df.pivot_table(values = '(kW)', index = 'YEAR', columns = 'Make', aggfunc = np.mean)

#list of functions to apply to the pivot table

df.pivot_table(values = '(kW)', index = 'YEAR', columns = 'Make', aggfunc = [np.mean, np.min], margins = True)


"""***DATE FUNCTIONALITY***"""

"""Time stamp"""

import pandas as pd
import numpy as np

pd.Timestamp('9/1/2016 10:05AM')

"""Period"""

#period month
pd.Period('1/2016')

#period day
pd.Period('3/5/2016')


"""Date time index"""

t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'),pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')]) 

type(t1.index) #datetimeindex

"""Period Index"""

t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])

type(t2.index) #periodindex

"""Converting to datetime"""

d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']

ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns = list('ab'))

ts3.index = pd.to_datetime(ts3.index)

pd.to_datetime('4.7.12', dayfirst=True)

"""Timedeltas"""
#differences in time

pd.Timestamp('9/3/2016') - pd.Timestamp('9/1/2016')

pd.Timestamp('9/2/2016') + pd.Timedelta('12D 3H')

"""Dates in a dataframe"""

#9 measurements taken biweekly, every sunday starting Oct 2016

dates = pd.date_range('10-01-2016', periods = 9, freq = '2W-SUN') #datetime index created

df = pd.DataFrame({'Count 1': 100 + np.random.randint(-5, 10, 9).cumsum(), 'Count 2': 120 + np.random.randint(-5, 10, 9)}, index = dates)

df.index.weekday_name #which day of each week is it

df.diff() #difference b/w each dates value

df.resample('M').mean() #mean count for each month in data frame

#partial string indexing to find value in a given year/month

df['2017'] #year

df['2016-12'] #month

df['2016-12':] #Slicing Dec 2016 onwards

df[:'2016-11'] #before Nov 2016

#change frequency of dates in df

df.asfreq('W', method = 'ffill') #missing values for every blank week, so forward fill used to fill the missing values


"""Plotting timeseries"""

import matplotlib.pyplot as plt

%matplotlib inline

df.plot()