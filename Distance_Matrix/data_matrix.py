# Data science Project Assignment 1
import pandas as pd
import numpy as np
# Loading the dataset
Datas= pd.read_csv('bank.csv',sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
#choosing Random 11% dataSet as a sample
# short_data = Datas.sample(frac=0.11, random_state=42)
short_data = Datas.tail(11)

#Displaying  the first 11  rows
# print(short_data)
# replacing the null value with 'NA' if  attributes consists any null values
replace_null = short_data.fillna('NA')
## Removing  rows with missing nominal attributes
# Identify nominal attributes (categorical columns)
nominal_columns = short_data.select_dtypes(include=['object']).columns
# display nominal attributes
# print("The nominal columns " , nominal_columns)
# Dropping rows with missing values in nominal columns
short_data.dropna(subset=nominal_columns, inplace=True)
# Displaying the clean dataSet
# print(" The  clean data set" , short_data)
# Identifying  numerical columns
numerical_columns = short_data.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical columns " , numerical_columns)
# Converting the numerical columns into data types
for column in numerical_columns:
    short_data[column] = pd.to_numeric(short_data[column], errors='coerce')
#printing the numerical column types and numerical datas
    print(short_data[numerical_columns])
    print(short_data.dtypes)
# Finding  any missing values among the numerical data types and replace with 0.00
short_data[numerical_columns] = short_data[numerical_columns].fillna(0.00)
# Normalize each numerical column
for column in numerical_columns:
    min_value = short_data[column].min()
    max_value = short_data[column].max()
    short_data[column] = (short_data[column] - min_value) / (max_value - min_value)

    print( "The normalization is " ,short_data[column])
