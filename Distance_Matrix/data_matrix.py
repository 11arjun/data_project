# Data science Project Assignment 1
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import hamming

# Loading the dataset
Datas= pd.read_csv('bank.csv',sep=';')
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.max_rows', None)     # Shows all rows
pd.set_option('display.max_colwidth', None) # Shows full column content
#choosing Random 0.5% dataSet as a sample
short_data = Datas.sample(frac=0.005, random_state=42)
#Displaying  datas
# print(short_data)
# replacing the null value with 'NA' if  attributes consists any null values
replace_null = short_data.fillna('NA')
# Displaying the clean dataSet
# print(" The  clean data set" , short_data)
# Identify nominal attributes (categorical columns)
nominal_columns = short_data.select_dtypes(include=['object']).columns
# Dropping rows with missing values in nominal columns
short_data.dropna(subset=nominal_columns, inplace=True)
# Setting the nominal attributes.
nominal_attributes = nominal_columns
#Displaying with the nominal datas
nominal_datas =  short_data[nominal_attributes]
# display nominal datas
# print("Nominal Datas \n" , nominal_datas)
# Encoding nominal Attributes
label_encoders = {}
encoded_nominal_data = short_data[nominal_columns].copy()  # Copy the relevant data for encoding
for column in nominal_attributes:
    le = LabelEncoder()
    encoded_nominal_data[column] = le.fit_transform(short_data[column])
    label_encoders[column] = le
# Converting  the encoded nominal data to numpy array for similarity calculation
nominal_encoded_data = encoded_nominal_data.values
print("\nEncoded Nominal Data:\n", nominal_encoded_data)
# Calculating similarity for nominal attributes using broadcasting
 #Create an empty similarity matrix
similarity_matrix = np.zeros((nominal_encoded_data.shape[0], nominal_encoded_data.shape[0]))
# Broadcasting comparison
for i in range(nominal_encoded_data.shape[0]):
    similarity_matrix[i, :] = np.mean(nominal_encoded_data == nominal_encoded_data[i], axis=1)
# Convert the similarity matrix back to a DataFrame for easier viewing
nominal_similarity_df = pd.DataFrame(similarity_matrix, index=short_data.index, columns=short_data.index)
#displaying the Similarity Matrix for Nominal Attributes
print("\nSimilarity Matrix for Nominal Attributes (Label Encoded):\n" , nominal_similarity_df)
# Identifying  numerical columns
numerical_columns = short_data.select_dtypes(include=[np.number]).columns.tolist()
# print("Continuous  Attributes \n ",  numerical_columns)
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
# print("continuious Normalizied ",continiuous_normalized)
# print(short_data)
# calculating the manhattan distance for each continuous Attributes
Manhattan_distance = manhattan_distances(continiuous_normalized)
print("\n Manhattan Distance Matrix for Continuous Attributes:")
print(Manhattan_distance)
# Inverting  the Manhattan distance to represent similarity
manhattan_similarity = 1 - Manhattan_distance / Manhattan_distance.max()
# Combining similarity of all attributes
combined_similarity = (manhattan_similarity + nominal_similarity_df.values ) / 2
print("\n Combined Similarity Matrix: \n" , combined_similarity)
# Convert combined_sim to a DataFrame
combined_similarity_df = pd.DataFrame(combined_similarity, index=short_data.index, columns=short_data.index)
#Writing the combined similarity matrix to a CSV file
combined_similarity_df.to_csv('combined_similarity_matrix.csv', index=False)
print("\nCombined Similarity Matrix has been written to 'combined_similarity_matrix.csv'")