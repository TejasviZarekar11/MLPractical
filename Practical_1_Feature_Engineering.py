#1.Apply basic feature engineering techniques such as handling missing values, encoding categorical variables, 
# and scaling numerical attributes on the suitable open access dataset to prepare a clean and model ready dataset
# for machine learning analysis.


# Import libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create dataset
data = {
    'Age': [20, 21, np.nan, 22, 23, 21],
    'Marks': [85, 90, 78, np.nan, 69, 84],
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Female'],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Pune', 'Delhi', 'Mumbai']
}

# Create DataFrame
df = pd.DataFrame(data)

print("Original data:\n", df)

print("\nMissing values:\n", df.isnull().sum())

# Handling missing values
imputer = SimpleImputer(strategy='mean')

df[['Age', 'Marks']] = imputer.fit_transform(
    df[['Age', 'Marks']]
)

print("\nAfter handling missing values:\n", df)

# Encoding categorical variables
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['City'] = le.fit_transform(df['City'])

print("\nAfter Encoding:", df)

# Scaling numerical attributes (Feature Scaling)
scaler = StandardScaler()

df[['Age', 'Marks']] = scaler.fit_transform(
    df[['Age', 'Marks']]
)

print("\nFinal Clean Dataset:\n", df.round(2))

print("\nMissing values:\n", df.isnull().sum())
