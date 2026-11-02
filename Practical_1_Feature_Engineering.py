#1.Apply basic feature engineering techniques such as handling missing values, encoding categorical variables, 
# and scaling numerical attributes on the suitable open access dataset to prepare a clean and model ready dataset
# for machine learning analysis.

```
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
```

Cell 2: Create Dataset

```
data = {
    'Age': [22, 38, np.nan, 35, 28, 45, np.nan, 30],
    'Fare': [7.25, 71.83, 8.05, np.nan, 13.0, 26.0, 15.5, 20.0],
    'Gender': ['Male', 'Female', 'Female', 'Male',
               np.nan, 'Male', 'Female', 'Male'],
    'Embarked': ['S', 'C', 'S', 'S', 'C',
                 np.nan, 'S', 'C'],
    'Survived': [0, 1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
```

Cell 3: Check Missing Values

```
print("Missing Values:")
print(df.isnull().sum())
```

Cell 4: Handle Missing Values

```
# Numerical columns
num_cols = ['Age', 'Fare']

# Categorical columns
cat_cols = ['Gender', 'Embarked']

# Fill numerical missing values with mean
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill categorical missing values with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("Dataset After Handling Missing Values:")
print(df)
```

Cell 5: Encode Categorical Variables

```
# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_data = encoder.fit_transform(df[cat_cols])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(cat_cols)
)

# Remove original categorical columns
df = df.drop(columns=cat_cols)

# Combine encoded columns
df = pd.concat([df, encoded_df], axis=1)

print("Dataset After Encoding:")
print(df)
```

Cell 6: Scale Numerical Attributes

```
scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

print("Dataset After Feature Scaling:")
print(df)
```

Cell 7: Final Dataset Verification

```
print("Final Clean Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)
```
