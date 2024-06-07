# Data science Project Assignment 1
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import LabelEncoder

# Loading the dataset
Datas= pd.read_csv('bank.csv',sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
#choosing Random 11% dataSet as a sample
short_data = Datas.sample(frac=0.1, random_state=42)
#Displaying  datas
# print(short_data)
# replacing the null value with 'NA' if  attributes consists any null values
replace_null = short_data.fillna('NA')
# Identify nominal attributes (categorical columns)
# Displaying the clean dataSet
# print(" The  clean data set" , short_data)
nominal_columns = short_data.select_dtypes(include=['object']).columns
# Dropping rows with missing values in nominal columns
short_data.dropna(subset=nominal_columns, inplace=True)
# Setting the nominal attributes.
nominal_attributes = nominal_columns
# display nominal attributes
print("Nominal Attributes" , nominal_attributes)
# Encoding nominal Attributes
label_encoders = {}
for column in nominal_attributes:
    le = LabelEncoder()
    short_data[column] = le.fit_transform(short_data[column])
    label_encoders[column] = le
# Displaying encoded nominal values
print("Encoded nominal values \n ",short_data[nominal_attributes])

# Calculating similarity  fro nominal attributes using Hamming distance  (one-hot encoded)
nominal_similarity = 
# Identifying  numerical columns
numerical_columns = short_data.select_dtypes(include=[np.number]).columns.tolist()
print("Continuous  Attributes \n ",  numerical_columns)
# Converting the numerical columns into data types
for column in numerical_columns:
    short_data[column] = pd.to_numeric(short_data[column], errors='coerce')
#printing the numerical column types and numerical datas
# print(short_data[numerical_columns])
# print(short_data.dtypes)
# Finding  any missing values among the numerical data types and replace with 0.00
short_data[numerical_columns] = short_data[numerical_columns].fillna(0.00)
# Normalizing each continuous attribute between [0-1]
for column in numerical_columns:
    min_value = short_data[column].min()
    max_value = short_data[column].max()
    short_data[column] = (short_data[column] - min_value) / (max_value - min_value)
#displaying continuous normalized dataset
continiuous_normalized = short_data[numerical_columns]
# print(continiuous_normalized)
# print(short_data)
# calculating the manhattan distance for each continuous Attributes
Manhattan_distance = manhattan_distances(continiuous_normalized)
print("\n Manhattan Distance Matrix for Continuous Attributes:")
print(Manhattan_distance)