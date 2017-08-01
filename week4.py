"""
Created on Tue Jan 31 14:09:18 2017

@author: fractaluser
"""

"""Distribution"""

#numpy built in distributions

import pandas as pd
import numpy as np



#In 20 coin flips how many achieve heads more than 15 times (iterate 10000 times)
np.random.binomial(1, 0.5)

np.random.binomial(1000, 0.5)/1000
              
check_ser = pd.Series()
for i in range(1,10000):
    check_ser.set_value(i, int(np.random.binomial(20, 0.5)>15))
    
    
check_ser.sum()/10000

#Better way
x = np.random.binomial(20, .5, 10000)
print((x>=15).mean())


#chance of a tornado in Ann Arbor

chance_of_tornado = 0.01/100

np.random.binomial(100000, chance_of_tornado)

chance_of_tornado = 0.01

#tornado happening two days in a row
tornado_events =  np.random.binomial(1, chance_of_tornado, 10000)

two_days_in_a_row = 0 #empty list

for j in range(1, len(tornado_events)-1):
    if tornado_events[j] == 1 and tornado_events[j-1] == 1:
        two_days_in_a_row += 1
        
print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))



#Continuous distribution

np.random.uniform(0,1)

#Normal distribution

np.random.normal(0.75)

distribution = np.random.normal(0.75, size = 1000)

#std of above distribution
np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))

#built in fn
np.std(distribution)

import scipy.stats as stats

#shape of the tails of the distribution(kurtosis)

# -ve value : curve slightly more flat than normal distribution
# +ve value : curve slightly more peaky than normal distribution
stats.kurtosis(distribution) #kurtosis of the 1000 values we sample out of the distribution

#Skewed distribution

#Chi Squared Distribution
#only one parameter, dof
# dof closely depends on the number of parameters you take from a normal population

#as dof increases the shape of the chi squared dist changes

chi_squared_df2 = np.random.chisquare(2, size = 10000)

stats.skew(chi_squared_df2) #quite a large skew

#change dof to 5

chi_squared_df5 =np.random.chisquare(5, size = 10000)

stats.skew(chi_squared_df5) #smaller skew


"""Hypothesis Testing"""


df = pd.read_csv('/home/fractaluser/Desktop/Documents/Coursera/Intro to data science with python/course1_downloads/grades.csv')

df.head()

len(df)

early = df[df['assignment1_submission'] <= '2015-12-31']

late = df[df['assignment1_submission'] > '2015-12-31']

early.mean()

late.mean()

#check out wiki page for t-test

#Most statistical tests expect that the data conforms to a certain distribution, a shape. So don't apply these tests blindly and investigate data first

from scipy import stats
#stats.ttest_ind?

stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade'])

#p value much higher than our 0.05 threshold, so we cannot reject the Null Hypothesis

#In lay terms: No statistically significant difference in these two populations

stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade']) #much larger again

stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade']) #larger than 0.05

#stats.ttest_ind(early['assignment6_grade'], late['assignment6_grade'])

