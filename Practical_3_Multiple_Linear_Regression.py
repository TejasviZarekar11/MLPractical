# 3 Develop a multiple linear regression model using a suitable real-world open-access dataset and assess its predictive capability using regression evaluation metrics such as MSE, RMSE, and R2.
                                                                                                                                                                                           
Cell 1: Import Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

Cell 2: Load Dataset

```
data = load_diabetes()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['Target'] = data.target

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)
```

Cell 3: Separate Input and Output Variables

```
X = df.drop('Target', axis=1)

y = df['Target']

print("Independent Variables:")
print(X.head())

print("\nDependent Variable:")
print(y.head())
```

Cell 4: Split Dataset

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
```

Cell 5: Create and Train Model

```
model = LinearRegression()

model.fit(X_train, y_train)

print("Multiple Linear Regression Model Trained Successfully")
```

Cell 6: Make Predictions

```
y_pred = model.predict(X_test)

print("Actual Values:")
print(y_test.head(10).values)

print("\nPredicted Values:")
print(y_pred[:10])
```

Cell 7: Evaluate Model Performance

```
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)
```

Cell 8: Visualize Actual vs Predicted Values

```
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted Values")

plt.show()
```
