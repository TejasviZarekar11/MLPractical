"""
Practical 1: Feature Engineering (Missing Values, Encoding, Scaling)

Aim    : Apply basic feature engineering - handle missing values, encode categorical variables and scale numerical attributes - to prepare a clean, model-ready dataset.
Dataset: Titanic-style passenger data (small inline sample; real Titanic CSV link given in code)

Run    : python Practical_1_Feature_Engineering.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd

# Titanic-style sample (small, runs without internet).
# For the real dataset use this instead:
# df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = pd.DataFrame({
    "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 3, 2, 3, 1, 3, 2],
    "Sex": ["male", "female", "female", "female", "male", "male", "male", "female",
            "female", "male", "male", "female", "male", "female", "female"],
    "Age": [22, 38, 26, 35, np.nan, 54, 2, 27, 14, np.nan, 28, 4, 58, 20, np.nan],
    "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.07, 11.13, 30.07,
             7.85, 13.00, 16.70, 26.55, 8.05, 26.00],
    "Embarked": ["S", "C", "S", "S", np.nan, "S", "S", "S", "C", "S", "S", "S", "S",
                 np.nan, "S"],
    "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
})
print("Shape:", df.shape)
print(df.head(8))

# ======================================================================
# Step 2
# ======================================================================
# Step 1: find the missing values
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

# ======================================================================
# Step 3
# ======================================================================
# Step 2: handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())                    # numeric -> median
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])    # category -> mode
print("Missing values after cleaning:", df.isnull().sum().sum())
print(df)

# ======================================================================
# Step 4
# ======================================================================
# Step 3: encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})                   # label encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)  # one-hot
print(df.head(8))

# ======================================================================
# Step 5
# ======================================================================
# Step 4: scale numerical attributes
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = df.drop("Survived", axis=1)
y = df["Survived"]
num_cols = ["Age", "Fare"]

X_std = X.copy()
X_std[num_cols] = StandardScaler().fit_transform(X[num_cols])   # mean 0, std 1
X_mm = X.copy()
X_mm[num_cols] = MinMaxScaler().fit_transform(X[num_cols])      # range 0 to 1

print("After StandardScaler:")
print("mean:", X_std[num_cols].mean().round(2).to_dict())
print("std :", X_std[num_cols].std(ddof=0).round(2).to_dict())
print("\nAfter MinMaxScaler:")
print("min:", X_mm[num_cols].min().round(2).to_dict())
print("max:", X_mm[num_cols].max().round(2).to_dict())

# ======================================================================
# Step 6
# ======================================================================
# Step 5: final model-ready dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "  Test:", X_test.shape)
print(X_std.head())
