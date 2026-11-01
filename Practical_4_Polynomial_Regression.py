"""
Practical 4: Polynomial Regression and the Bias-Variance Trade-off

Aim    : Demonstrate polynomial regression with varying degrees and observe the effect of model complexity on accuracy (underfitting vs overfitting).
Dataset: Diabetes dataset (sklearn) - BMI feature vs disease progression

Run    : python Practical_4_Polynomial_Regression.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True, as_frame=True)
X = X[["bmi"]]                                  # one feature, so we can plot the curve
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=40, random_state=3)         # small training set shows overfitting
print("Train size:", len(X_train), " Test size:", len(X_test))

# ======================================================================
# Step 2
# ======================================================================
# Fit polynomial models of different degrees
degrees = [1, 2, 3, 5, 8, 12]
rows = []
models = {}
for d in degrees:
    model = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())
    model.fit(X_train, y_train)
    models[d] = model
    tr_mse = mean_squared_error(y_train, model.predict(X_train))
    te_mse = mean_squared_error(y_test, model.predict(X_test))
    te_r2 = r2_score(y_test, model.predict(X_test))
    rows.append([d, round(tr_mse, 1), round(te_mse, 1), round(te_r2, 3)])

print(pd.DataFrame(rows, columns=["Degree", "Train MSE", "Test MSE", "Test R2"]))

# ======================================================================
# Step 3
# ======================================================================
# Plot the fitted curves
x_line = pd.DataFrame({"bmi": np.linspace(X["bmi"].min(), X["bmi"].max(), 200)})

plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, color="black", s=20, label="Train data")
for d, c in zip([1, 3, 12], ["blue", "green", "red"]):
    plt.plot(x_line["bmi"], models[d].predict(x_line), color=c, label=f"Degree {d}")
plt.ylim(y.min() - 20, y.max() + 20)
plt.xlabel("BMI (scaled)"); plt.ylabel("Disease progression")
plt.title("Polynomial Regression - different degrees")
plt.legend()
plt.show()

# ======================================================================
# Step 4
# ======================================================================
# Bias-variance: train error vs test error
tr = [r[1] for r in rows]
te = [r[2] for r in rows]

plt.figure(figsize=(7, 4))
plt.plot(degrees, tr, marker="o", label="Train MSE")
plt.plot(degrees, te, marker="o", label="Test MSE")
plt.yscale("log")                      # log scale: very high degrees blow up
plt.xlabel("Polynomial degree (model complexity)")
plt.ylabel("MSE (log scale)")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid(True)
plt.show()

best = degrees[int(np.argmin(te))]
print("Best degree (lowest test MSE):", best)
