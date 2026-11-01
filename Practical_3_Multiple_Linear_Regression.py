"""
Practical 3: Multiple Linear Regression

Aim    : Develop a multiple linear regression model on a real-world dataset and assess it using MSE, RMSE and R2.
Dataset: Diabetes dataset (sklearn, 442 patients, 10 features, target = disease progression)

Run    : python Practical_3_Multiple_Linear_Regression.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = load_diabetes(as_frame=True)
df = data.frame
print("Shape:", df.shape)
print(df.head().round(3))

# ======================================================================
# Step 2
# ======================================================================
# Split features (X) and target (y), then train / test split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ======================================================================
# Step 3
# ======================================================================
# Train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", round(model.intercept_, 2))
print("\nCoefficients:")
print(pd.Series(model.coef_, index=X.columns).round(2))

# ======================================================================
# Step 4
# ======================================================================
# Predict and evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("MAE  :", round(mae, 2))
print("R2   :", round(r2, 4))
print("Train R2:", round(model.score(X_train, y_train), 4))

# ======================================================================
# Step 5
# ======================================================================
# Actual vs predicted and residual plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(y_test, y_pred, alpha=0.7)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
ax[0].set_title("Actual vs Predicted")

ax[1].scatter(y_pred, y_test - y_pred, alpha=0.7)
ax[1].axhline(0, color="r", linestyle="--")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Residual")
ax[1].set_title("Residual Plot")
plt.tight_layout()
plt.show()
