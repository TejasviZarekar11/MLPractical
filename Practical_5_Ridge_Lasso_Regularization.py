# 5 Classify data using regularization techniques such as Ridge and Lasso regression on a suitable multivariate open-access dataset and observe the influence on model overfitting and multicollinearity.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

results = []
coefficients = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)

    results.append([name, train_mse, test_mse, r2])
    coefficients[name] = model.coef_

# Display results
df = pd.DataFrame(
    results,
    columns=["Model", "Train MSE", "Test MSE", "R2 Score"]
)

print(df)

# Graph 1: MSE comparison
df.plot(x="Model", y=["Train MSE", "Test MSE"], kind="bar")
plt.title("Training and Testing MSE Comparison")
plt.ylabel("MSE")
plt.show()

# Graph 2: Coefficient comparison
for name, coef in coefficients.items():
    plt.plot(coef, marker="o", label=name)

plt.title("Coefficient Comparison")
plt.xlabel("Features")
plt.ylabel("Coefficient Values")
plt.legend()
plt.show()

# Number of selected features by Lasso
print("Lasso Selected Features:",
      np.sum(coefficients["Lasso"] != 0))
