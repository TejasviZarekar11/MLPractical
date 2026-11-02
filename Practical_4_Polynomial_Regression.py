# 4 Demonstrate the use of polynomial regression with varying polynomial degrees on a suitable open-access dataset and observe the effect of model complexity on prediction accuracy in terms of the bias–variance trade-off.
 Cell 1: Import Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

Cell 2: Load Dataset

```
data = load_diabetes()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['Target'] = data.target

print(df.head())

print("Dataset Shape:", df.shape)
```

Cell 3: Select Input and Output Variables

```
X = df[['bmi']]

y = df['Target']

print("Input Feature: BMI")
print("Output Feature: Disease Progression")
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

Cell 5: Apply Polynomial Regression with Different Degrees

```
degrees = [1, 2, 3, 4, 5]

results = []

for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    test_r2 = r2_score(y_test, test_pred)

    results.append([
        degree, train_mse, test_mse, test_r2
    ])

results_df = pd.DataFrame(
    results,
    columns=['Degree', 'Training MSE', 'Testing MSE', 'Testing R2']
)

print(results_df)
```

Cell 6: Visualize Different Polynomial Degrees

```
plt.figure(figsize=(10, 6))

plt.scatter(X, y, label='Actual Data')

X_curve = pd.DataFrame({
    'bmi': np.linspace(X['bmi'].min(), X['bmi'].max(), 200)
})

for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)
    X_curve_poly = poly.transform(X_curve)

    model = LinearRegression()

    model.fit(X_poly, y)

    y_curve = model.predict(X_curve_poly)

    plt.plot(
        X_curve,
        y_curve,
        label=f'Degree {degree}'
    )

plt.xlabel('BMI')
plt.ylabel('Disease Progression')

plt.title('Polynomial Regression with Different Degrees')

plt.legend()
plt.show()
```

Cell 7: Compare Training and Testing Errors

```
plt.plot(
    results_df['Degree'],
    results_df['Training MSE'],
    marker='o',
    label='Training MSE'
)

plt.plot(
    results_df['Degree'],
    results_df['Testing MSE'],
    marker='o',
    label='Testing MSE'
)

plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')

plt.title('Bias-Variance Trade-off')

plt.legend()
plt.show()
```
