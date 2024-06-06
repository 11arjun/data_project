# Data science Project Assignment 1
import pandas as pd
import numpy as np
# Loading the dataset
Datas= pd.read_csv('bank.csv',sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
#choosing Random 11% dataSet as a sample
short_data = Datas.sample(frac=0.11, random_state=42)

#Displaying  the first 11  rows
# print(short_data)
# replacing the null value with 'NA' if  attributes consists any null values
replace_null = short_data.fillna('NA')
## Removing  rows with missing nominal attributes
# Identify nominal attributes (categorical columns)
nominal_columns = short_data.select_dtypes(include=['object']).columns
# display nominal attributes
print("The nominal columns " + nominal_columns)
# Dropping rows with missing values in nominal columns
short_data.dropna(subset=nominal_columns, inplace=True)
# Displaying the clean dataSet
print(short_data)
# Identifying  numerical columns
numerical_columns = short_data.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical columns identified:")
print(numerical_columns)
# Convert the numerical columns into types