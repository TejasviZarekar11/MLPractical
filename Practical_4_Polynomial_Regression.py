# 4 Demonstrate the use of polynomial regression with varying polynomial degrees on a suitable open-access dataset and observe the effect of model complexity on prediction accuracy in terms of the bias–variance trade-off.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Real-World Dataset
data = load_diabetes(return_X_y=True, as_frame=True)

X = data[0][["bmi"]]
y = data[1]

print("Dataset Shape:", X.shape)

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=40, random_state=3
)

# 3. Different Polynomial Degrees
degrees = [1, 2, 3, 5, 8, 12]

results = []

for degree in degrees:

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    rmse = np.sqrt(test_mse)
    r2 = r2_score(y_test, test_pred)

    results.append([
        degree, train_mse, test_mse, rmse, r2
    ])

# 4. Display Results
df = pd.DataFrame(
    results,
    columns=["Degree", "Train MSE", "Test MSE", "RMSE", "R2 Score"]
)

print("\nPerformance Comparison:")
print(df)

# 5. Polynomial Regression Graph
plt.scatter(X, y, label="Original Data")

x_values = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

for degree in degrees:

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(x_values)

    plt.plot(x_values, y_pred, label=f"Degree {degree}")

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# 6. Bias-Variance Trade-off Graph
plt.plot(df["Degree"], df["Train MSE"], marker="o", label="Train MSE")

plt.plot(df["Degree"], df["Test MSE"], marker="o", label="Test MSE")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.show()
