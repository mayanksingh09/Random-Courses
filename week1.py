"""
Created on Mon Jan  2 16:20:42 2017

@author: mayank
"""

x = 1
y = 2

x + y


"""
can set default values for variables, like z

all the optional parameters at the end of the fn
"""

def add_number(x, y, z=None):
    if (z == None):
        return x + y
    else:
        return x + y + z
        
print(add_number(1,2))
print(add_number(1,2,4))
    
def add_number(x, y, z=None, flag = False):
    if (flag):
        print('Flag is true')
    if (z==None):
        return x + y 
    else:
        return x + y + z

print(add_number(1,2, flag = True))


"""
TYPES AND SEQUENCES
"""

type('hey')

type(1)

type(1.2)

type(None)

type(add_number)


"Tuple"

x = (1, 'a', 2, 'cas')

type(x)

"Lists"

x = [1,'a', 2, 'cas']

type(x)

x.append(3)

x
"Iterate through the list/tuple"
for item in x:
    print(item)
    
i = 0
while( i != len(x)):
    print(x[i])
    i = i + 1

y = [2, 1, 'ad', 'asd']

"+ concatenates lists"

x + y

"* repeats the values"
(x + y)*3

"check if item is present in list or tuple"

1 in x


"***IMPORTANT***"
"slicing in lists"

x = "This is a string"

print(x[0])

print(x[0:1])

print(x[0:4])

"can do it backwards as well"

print(x[-1])

print(x[-4:-1])

x[:3] 

x[3:]


firstname = "Mayank"
lastname = "Singh"
print(firstname + ' ' + lastname)

print(firstname*3)

print('May' in firstname)


"split function for strings"
"creates a simple list"


firstname = "Mayank Veer Singh".split(" ")[0]
lastname = "Mayank Veer Singh".split(" ")[-1]

print(firstname + " " + lastname)
 
 x = {'Mayank Singh': 'mayanksingh@gmail.com', 'Bill Clinton': 'billclinton@email.com'}
 
 x['Mayank Singh']
 
 x['ted'] = 'tedthestoner@mail.com'
 
 x['ted']
 
 "iterate using the keys"
 for name in x:
     print(x[name])
    
    
"iterate just the values"
for email in x.values():
    print(email)
    
    
"iterate over both using items fn"    
for name, email in x.items():
    print(name)
    print(email)
    
    
"Unpacking"
"in one statement"

x = ('Mayank', 'Singh', 'mayanksingh@gmail.com')
fname, lname, email = x

fname

lname

"More operations on strings"


"String formatting language"


sales_record = {'price': 3.12,
                'num_items':4,
                'person': 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))



"Reading and writing csv" 
"import library csv to load csvs"

import csv 

"limit decimal point to 2"
%precision 2

"opening csv and importing it"
with open('/home/fractaluser/Documents/Coursera/Intro to data science with python/course1_downloads/mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
    
    
mpg[:3]

len(mpg)

"checking column names"
mpg[0].keys()

"avg city mpg across all cars in the file"
sum(float(d['cty']) for d in mpg) / len(mpg)

"highway mpg"
sum(float(d['hwy']) for d in mpg) / len(mpg)

"avg mpg grouped by no. of cylinders"
cylinders  = set(d['cyl'] for d in mpg)
cylinders

"empty list to store calculations"
ctympgbycyl = []

"iterate over all cylinder levels then, iterate over all dictionaries. If the cylinder level for the dictionary we are on matches the cylider level we are finding the average of we add the mpg to the summpg variable and increment the count. After going through all the dictionaries we perform the avg mpg calc and then append it to our list"

for c in cylinders:
    summpg = 0
    cyltypecount = 0
    for d in mpg:
        if d['cyl'] == c:
            summpg += float(d['cty'])
            cyltypecount += 1
    ctympgbycyl.append((c, summpg / cyltypecount))
    
ctympgbycyl.sort(key=lambda x: x[0])

ctympgbycyl

"avg hwy mpg for different vehicle classes"
vehicleclass  = set(d['class'] for d in mpg)

hwympgbyclass = []

for v in vehicleclass:
    sumhmpg = 0
    vehtypecount = 0
    for d in mpg:
        if d['class'] == v:
            sumhmpg += float(d['hwy'])
            vehtypecount += 1
    hwympgbyclass.append((v, sumhmpg / vehtypecount))

hwympgbyclass.sort(key=lambda x:x[1])



"***DATE & TIME***"
import datetime as dt
import time as tm

"Current time"
tm.time()

"Creating time stamp"
dtnow = dt.datetime.fromtimestamp(tm.time())

dtnow

"attributes of datetime fn"
dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.second

"use timedelta to carry out simple math operations on dates"
delta = dt.timedelta(100)

today = dt.date.today()

today-delta

today > today-delta



"***ADV PYTHON OBJECTS***"

"Creating classes in python"
"usually declared as sentence case"
class Person:
    department = "School of Information"
    def set_name(self, new_name): #to define a method you just write it as you would have a fn, include self in the method signature
        self.name = new_name #referring to instance variables set on the object
    def set_location(self, new_location):
        self.location = new_location

#objects in python don't have private or protected members

#there is no need for an explicit constructer when creating objects in python

"map() fn, basis of functional programming in Python"
#Fn programming is the programming paradigm in which you explicitly declare all parameters which could change through execution of this fn.

#map(function, iterable,...)

store1 = [10.00,11.00, 12.34, 2.54]
store2 = [9.00, 11.10, 12.34, 2.01]

cheapest = map(min, store1, store2)

cheapest #lazy evaluation, map fn returns a map object

#maps are iterable like lists and tuples

"lambda fns"
my_fn = lambda a, b, c: a + b + c 
#declarea a lambda with lambda followed by list of arguments, then : and then a single expression. only one expression

my_fn(1,2,1)

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]
    
    
#using map()
list(map(split_title_and_name, people))

lmd = lambda person: person.split()[0] + ' ' + person.split()[-1]

list(map(lmd, people))


#using for loop
for person in people:
    print(split_title_and_name(person) == (lambda x: x.split()[0] + ' ' + x.split()[-1])(person))
    

"List Comprehension"
#create lists/tuples/dictionaries with abbreviated syntax

#using for loop
my_list = []
for number in range(0,1000):
    if number % 2 == 0:
        my_list.append(number)
        
my_list

#using list comprehension, faster and compact
my_list2 = [number for number in range(0,1000) if number % 2 == 0]

my_list2

#for loop
def times_tables():
    lst = []
    for i in range(10):
        for j in range (10):
            lst.append(i*j)
    return lst

#list comprehension
timetables2 = [j*i for i in range(10) for j in range(10)]


#assignment
lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'

lowercase[0]

for i in range(len(lowercase)+1):
    print(i)


for i in range(len(lowercase)):
        first = lowercase[i]
        print(first)
        for j in range(len(lowercase)):
            second = lowercase[j]
            print(first + second)

#for loop
def custid():
    lst = []
    for i in range(len(lowercase)):
        first = lowercase[i]
        for j in range(len(lowercase)):
            second = lowercase[j]
            for k in range(len(digits)):
                third = digits[k]
                for l in range(len(digits)):
                    fourth = digits[l]
                    lst.append(first+second+third+fourth)
    return lst

custid()

#list comprehensions
correct_answer = [a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]



"****NUMPY****

import numpy as np

mylist = [1,2,3]
x = np.array(mylist)

y = np.array([2,13,41,22])

#Multidimensional array, by passing a list in list
m = np.array([[1,2,31,33], [12,31,2,3]])

#check dimensions
m.shape

#arange() fn

n = np.arange(0,30,2) #start, stop, step-size

#convert to 3X5 array
n.reshape(3,5)

#linspace() fn
o = np.linspace(0,4,9) #start, stop, number of numbers

o.reshape(3,3) #doesn't assign new shape to o
o.resize(3,3) #changes the shape and assigns it to o

np.ones((3,3)) #array of ones

np.zeros((2,3))

np.eye(3) #identity matrix

np.diag(y) #creates diagonal array

np.array([2,31,2]*3)

np.repeat([2,31,2],3) #repeats each element of list 3 times before moving on the next

p = np.ones([2,3], int)

#stack it vertically
np.vstack([p, 2*p])

#horizontally
np.hstack([p,2*p])

a = np.array([2,3,1,2])
b = np.array([21,321,33,2])

a + b

a * b

#dot product
a.dot(b)

z = np.array([y, y**2])

#transpose
z.T

#check the data the array has
z.dtype

z = z.astype('f')

z.dtype

a = np.array([-4,-2,-4,2,3])

a.sum()

a.max()

a.min()

a.mean()

a.std()

#index/position of max value
a.argmax()

#index/position of min value
a.argmin()

"INDEXING AND SLICING"

#square is ** not ^
a = np.arange(13)**2

a[5] #5th digit

#start:end:step-size
a[4:10:2] #range

a[-4:] #last 4 digits

a[-5::2]

r = np.arange(36)
r.resize(6,6)

r[2,2] #2nd row 2nd column element

r[3, 3:6] #3rd row 3rd to 6th column elements

r[:2, :-1] #till second row and all column except last

r[-1,::2] #every second element from the last row

#all values greater than 30 capped to 30
r[r>30] = 30

#slice r
r2 = r[:3,:3]

#that part of r is also changed
r2[:] = 0

#now r2_copy changes wont change r
r2_copy = r.copy()

r2_copy[:] = 0

"ITERATE OVER ARRAYS"

test = np.random.randint(0,10,(4,3))

#iterating by row
for row in test:
    print(row)
    
#iterating by row index, len(test) gives no of rows
for i in range(len(test)):
    print(test[i])

#enumerate() gives row and index of row 

for i, row in enumerate(test):
    print('row',i,'is',row)
    
test2 = test**2

#zip(), iterate through both arrays

for i, j in zip(test,test2):
    print(i, '+', j, '=', i+j)