"""
Practical 5: Ridge and Lasso Regularization

Aim    : Use Ridge and Lasso regularization on a multivariate dataset and observe their influence on overfitting and multicollinearity (regression and classification).
Dataset: Diabetes dataset (regression) and Breast Cancer dataset (classification) from sklearn

Run    : python Practical_5_Ridge_Lasso_Regularization.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

X, y = load_diabetes(return_X_y=True, as_frame=True)

# Multicollinearity check: s1 and s2 are highly correlated
print("Correlation(s1, s2) =", round(X["s1"].corr(X["s2"]), 2))
plt.figure(figsize=(7, 5))
sns.heatmap(X.corr(), annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Feature correlation")
plt.show()

# ======================================================================
# Step 2
# ======================================================================
# Compare Linear, Ridge and Lasso
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

models = {"Linear": LinearRegression(),
          "Ridge": Ridge(alpha=0.1),
          "Lasso": Lasso(alpha=0.3)}
coefs = pd.DataFrame(index=X.columns)

for name, m in models.items():
    m.fit(X_train, y_train)
    coefs[name] = m.coef_
    print(f"{name:7s} Train R2 = {m.score(X_train, y_train):.3f}   "
          f"Test R2 = {m.score(X_test, y_test):.3f}")

print("\nCoefficients:")
print(coefs.round(1))

# ======================================================================
# Step 3
# ======================================================================
# Coefficient comparison: Ridge shrinks, Lasso sets some to exactly zero
coefs.plot(kind="bar", figsize=(9, 4))
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Coefficients: Linear vs Ridge vs Lasso")
plt.ylabel("Coefficient value")
plt.tight_layout()
plt.show()

# ======================================================================
# Step 4
# ======================================================================
# Effect of alpha (regularization strength)
alphas = np.logspace(-3, 1, 30)
ridge_r2, lasso_r2, lasso_nz = [], [], []
for a in alphas:
    ridge_r2.append(Ridge(alpha=a).fit(X_train, y_train).score(X_test, y_test))
    las = Lasso(alpha=a, max_iter=10000).fit(X_train, y_train)
    lasso_r2.append(las.score(X_test, y_test))
    lasso_nz.append(np.sum(las.coef_ != 0))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogx(alphas, ridge_r2, label="Ridge")
ax[0].semilogx(alphas, lasso_r2, label="Lasso")
ax[0].set_xlabel("alpha"); ax[0].set_ylabel("Test R2"); ax[0].legend()
ax[0].set_title("Test R2 vs alpha")
ax[1].semilogx(alphas, lasso_nz, color="green")
ax[1].set_xlabel("alpha"); ax[1].set_ylabel("Non-zero coefficients")
ax[1].set_title("Lasso feature selection")
plt.tight_layout()
plt.show()

# ======================================================================
# Step 5
# ======================================================================
# Choose the best alpha automatically with cross-validation
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = RidgeCV(alphas=np.logspace(-3, 2, 50)).fit(X_train, y_train)
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_train, y_train)
print("Best Ridge alpha:", round(ridge_cv.alpha_, 4),
      "| Test R2:", round(ridge_cv.score(X_test, y_test), 3))
print("Best Lasso alpha:", round(lasso_cv.alpha_, 4),
      "| Test R2:", round(lasso_cv.score(X_test, y_test), 3))
print("Features kept by Lasso:", list(X.columns[lasso_cv.coef_ != 0]))

# ======================================================================
# Step 6
# ======================================================================
# Overfitting demo: 65 polynomial features but only 60 training rows
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=60, random_state=1)
steps = lambda reg: make_pipeline(PolynomialFeatures(2), StandardScaler(), reg)

for name, reg in [("Linear", LinearRegression()),
                  ("Ridge", Ridge(alpha=100)),
                  ("Lasso", Lasso(alpha=2, max_iter=50000))]:
    m = steps(reg).fit(X_tr, y_tr)
    print(f"{name:7s} Train R2 = {m.score(X_tr, y_tr):.3f}   Test R2 = {m.score(X_te, y_te):.3f}")

# ======================================================================
# Step 7
# ======================================================================
# Classification with regularization: Breast Cancer data
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, RidgeClassifier

Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=42, stratify=yc)

clfs = {
    "Ridge classifier": RidgeClassifier(alpha=10),
    "Logistic (L2 / Ridge)": LogisticRegression(penalty="l2", C=1, max_iter=5000),
    "Logistic (L1 / Lasso)": LogisticRegression(penalty="l1", solver="liblinear", C=0.1),
}
for name, clf in clfs.items():
    pipe = make_pipeline(StandardScaler(), clf).fit(Xc_tr, yc_tr)
    zero = (clf.coef_ == 0).sum()
    print(f"{name:24s} Test accuracy = {pipe.score(Xc_te, yc_te):.3f}  "
          f"| zero coefficients = {zero} of {Xc.shape[1]}")
