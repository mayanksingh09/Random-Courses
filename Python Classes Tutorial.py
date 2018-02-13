#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:36:29 2018

"""

## Class
            
class Employee:
    
    num_of_employees = 0
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): # initializing class variable
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
        Employee.num_of_employees += 1 # counting the number of employees
        
        
    # creating a method to display full name
    def fullname(self):
        print('{} {}'.format(self.first, self.last))
        
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)
    
emp_1 = Employee('Mayank', 'Singh', 150000) # an instance of the class employee
emp_2 = Employee('Hey', 'There', 20000)

emp_1.fullname()
Employee.fullname(emp_1) # running the method from the class, same result as above

# Class variables same for each instance across the class


Employee.raise_amount = 1.05 # changes the value for the class (and all its instances)
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

emp_1.raise_amount = 1.08 # changes the value only for this instance after creating that arribute for employee 1

print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)


print(emp_1.__dict__) # name space(dictionary) of emp_1
print(Employee.__dict__) # name space of the class


print(Employee.num_of_employees) # number of employees after instantiating emp_1 and emp_2


## CLASS METHODS

# Class methods (fullname, apply_raise were regular methods taking in the instance as the first argument)


class Employee:
    
    num_of_employees = 0
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): # initializing class variable
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
        Employee.num_of_employees += 1 # counting the number of employees
        
        
    # creating a method to display full name
    def fullname(self):
        print('{} {}'.format(self.first, self.last))
        
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

    @classmethod # receive class as first argument instead of instance
    def set_raise_amt(cls, amount): # common convention like self for class methods is cls
        cls.raise_amount = amount
        
        
emp_1 = Employee('Mayank', 'Singh', 150000) # an instance of the class employee
emp_2 = Employee('Hey', 'There', 20000)

Employee.set_raise_amt(1.09) # using class methods to change the value of raise_amount using the amount input in set_raise_amt 

emp_1.set_raise_amt(1.11) # can also run it from instance (still changes it for the class so also all the instances, makes little sense to use this)

# all change to 1.11
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

## Use class methods as alternative constructors (Use class methods to provide ways to create objects)

emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Mr-Smith-1000000'
emp_str_3 = 'Jay-Sean-2000'

# Create a constructor that accepts the above string and creates employees for them

class Employee:
    
    num_of_employees = 0
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): # initializing class variable
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
        Employee.num_of_employees += 1 # counting the number of employees
        
        
    # creating a method to display full name
    def fullname(self):
        print('{} {}'.format(self.first, self.last))
        
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

    @classmethod # receive class as first argument instead of instance
    def set_raise_amt(cls, amount): # common convention like self for class methods is cls
        cls.raise_amount = amount
    
    @classmethod
    def from_string(cls, emp_str): # splits the string to crate name and pay
        first, last, pay = emp_str.split('-')
        return  cls(first, last, pay)
    
new_emp_1 = Employee.from_string(emp_str_1) #from_string alternative constructor

print(new_emp_1.email)
print(new_emp_1.pay) 


## STATIC METHODS

# Don't pass anything automatically (instances or classes)

# Function to check if it was a workday or not

class Employee:
    
    num_of_employees = 0
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): # initializing class variable
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
        Employee.num_of_employees += 1 # counting the number of employees
        
        
    # creating a method to display full name
    def fullname(self):
        print('{} {}'.format(self.first, self.last))
        
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

    @classmethod # receive class as first argument instead of instance
    def set_raise_amt(cls, amount): # common convention like self for class methods is cls
        cls.raise_amount = amount
    
    @classmethod
    def from_string(cls, emp_str): # splits the string to crate name and pay
        first, last, pay = emp_str.split('-')
        return  cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if (day.weekday() == 5) or (day.weekday() == 6):
            return False
        return True

import datetime
my_date = datetime.date(2016, 7, 11)    

print(Employee.is_workday(my_date)) # check if its a workday


## Class Inheritance

class Employee:
    
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): # initializing class variable
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
        
        
    # creating a method to display full name
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
        
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)


dev_1 = Employee('Mayank', 'Singh', 67131241)
dev_2 = Employee('Yuvraj', 'Singh', 123441155)

print(dev_1.email)
print(dev_2.email)

## creating developer and manager subclasses

class Developer(Employee):
    raise_amt = 1.10 # can change the raise amount in the subclass (will not change fro the employee class)
    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay) # letting the employee class handle the first, last and pay arguments
        # Employee.__init__(self, first, last, pay) # same as super()... above
        self.prog_lang = prog_lang
        

dev_1 = Developer('Mayank', 'Singh', 50000000, 'Python')
dev_2 = Developer('Yuvraj', 'Singh', 2000000, 'R Programming')

print(help(Developer)) # method resolution order and other information printed out

print(dev_1.email) # can access attributes set in parent Employee class
print(dev_1.prog_lang)
print(dev_2.email)
print(dev_2.prog_lang)

print(dev_1.pay)
dev_1.apply_raise() #applied raise from the parent employee class
print(dev_1.pay)


## creating another subclass Manager

class Manager(Employee):
    raise_amt = 1.2
    def __init__(self, first, last, pay, experience, employees = None): # a list of employees
        super().__init__(first, last, pay)
        self.experience = experience
        if employees is None:
            self.employees  = []
        else:
            self.employees = employees
            
    
    def add_emp(self, emp): # add an employee to the list
        if emp not in self.employees:
            self.employees.append(emp)
            
            
    def rem_emp(self, emp): # remove an employee from the list
        if emp in self.employees:
            self.employees.remove(emp)
            
    def print_emps(self): # print all employees in the list
        for emp in self.employees:
            print('-->', emp.fullname())
        
manager_1 = Manager('Antonio', 'Conte', 123000000, '4+ yrs', [dev_1])
manager_2 = Manager('Jose', 'Mourinho', 12340000, '8 yrs', ['Paul Pogba', 'Romelu Lukaku'])

print(manager_1.email)
print(manager_1.experience)
manager_1.print_emps()

manager_1.add_emp(dev_2) # removing a manager
manager_1.rem_emp(dev_1) # removing a developer

isinstance(manager_1, Manager) #checks if an object is an instance of a class
isinstance(manager_1, Employee) # an instance of Employee
isinstance(manager_1, Developer) # not an instance of Developer

issubclass(Manager, Employee) # checks if its a subclass of a class
issubclass(Developer, Employee)
issubclass(Developer, Manager)



## MAGIC METHODS/DUNDER init

# can be used to change build in python behaviour
#def __repr__(self): # unambiguous representation of objects
#    pass
#
#def __str__(self): # readable representation of an object
#    pass


class Employee:
    
    raise_amount = 1.04
    
    def __init__(self, first, last, pay): # initializing class variable
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
        
        
    # creating a method to display full name
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
        
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

    def __repr__(self): 
        return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)

    def __str__(self):
        return '{} - {}'.format(self.fullname(), self.email)


emp_1 = Employee('A', 'B', 1234)

print(emp_1)

print(repr(emp_1))
print(str(emp_1)) # calling the special methods

