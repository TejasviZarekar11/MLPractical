# 5 Classify data using regularization techniques such as Ridge and Lasso regression on a suitable multivariate open-access dataset and observe the influence on model overfitting and multicollinearity.
Cell 1: Import Libraries

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
```

Cell 2: Load Dataset

```
data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset Shape:", X.shape)
print(X.head())
```

Explanation: `load_diabetes()` loads the built-in dataset. `X` contains input features and `y` contains the target.
Cell 3: Split Dataset

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
```

Cell 4: Feature Scaling

```
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Important: Scaling is useful because Ridge and Lasso penalize coefficients. Without scaling, features with different scales may be penalized unfairly.
Cell 5: Train Linear Regression

```
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_train = lr.predict(X_train)
lr_test = lr.predict(X_test)
```

Cell 6: Train Ridge Regression

```
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

ridge_train = ridge.predict(X_train)
ridge_test = ridge.predict(X_test)
```

Important line: `alpha=1.0` controls the strength of regularization. Higher alpha means a stronger penalty.
Cell 7: Train Lasso Regression

```
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train, y_train)

lasso_train = lasso.predict(X_train)
lasso_test = lasso.predict(X_test)
```

Important line: Lasso can make some coefficients zero, which helps in feature selection.
Cell 8: Evaluate Model Performance

```
models = {
    "Linear": (lr_train, lr_test),
    "Ridge": (ridge_train, ridge_test),
    "Lasso": (lasso_train, lasso_test)
}

results = []

for name, (train_pred, test_pred) in models.items():

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    mse = mean_squared_error(y_test, test_pred)

    results.append([
        name, train_r2, test_r2, mse
    ])

result_df = pd.DataFrame(
    results,
    columns=["Model", "Training R2", "Testing R2", "MSE"]
)

print(result_df)
```


Cell 9: Compare Model Performance

```
result_df.set_index("Model")[["Training R2", "Testing R2"]].plot(
    kind="bar", figsize=(8, 5)
)

plt.title("Comparison of Regression Models")
plt.ylabel("R2 Score")
plt.show()
```

Cell 10: Compare Coefficients

```
coef_df = pd.DataFrame({
    "Feature": data.feature_names,
    "Linear": lr.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

print(coef_df)
```


Cell 11: Observe Multicollinearity

```
corr = pd.DataFrame(
    data.data,
    columns=data.feature_names
).corr()

plt.figure(figsize=(10, 7))

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.show()
```
