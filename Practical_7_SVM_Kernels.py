"""
Practical 7: Support Vector Machine - Linear and Non-linear Kernels

Aim    : Use SVM classifiers with linear and non-linear kernels and compare their performance using suitable evaluation metrics.
Dataset: Breast Cancer dataset (kernel comparison) and make_moons (visualising kernels)

Run    : python Practical_7_SVM_Kernels.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ======================================================================
# Step 2
# ======================================================================
# Compare kernels on Breast Cancer data
rows = []
for k in ["linear", "poly", "rbf", "sigmoid"]:
    svm = SVC(kernel=k, C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    p = svm.predict(X_test)
    rows.append([k, accuracy_score(y_test, p), precision_score(y_test, p),
                 recall_score(y_test, p), f1_score(y_test, p)])

print(pd.DataFrame(rows, columns=["Kernel", "Accuracy", "Precision", "Recall", "F1"]).round(4))

# ======================================================================
# Step 3
# ======================================================================
# Non-linear data: linear kernel vs RBF kernel
from sklearn.inspection import DecisionBoundaryDisplay

Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=42)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=42)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, k in zip(ax, ["linear", "rbf"]):
    svm = SVC(kernel=k, C=1.0).fit(Xm_tr, ym_tr)
    DecisionBoundaryDisplay.from_estimator(svm, Xm, ax=a, alpha=0.3, cmap="coolwarm")
    a.scatter(Xm_te[:, 0], Xm_te[:, 1], c=ym_te, cmap="coolwarm", edgecolor="k", s=25)
    a.set_title(f"{k} kernel - test accuracy = {svm.score(Xm_te, ym_te):.3f}")
plt.tight_layout()
plt.show()

# ======================================================================
# Step 4
# ======================================================================
# Effect of C and gamma for the RBF kernel (moons data)
rows = []
for C in [0.1, 1, 10]:
    for gamma in [0.1, 1, 10]:
        svm = SVC(kernel="rbf", C=C, gamma=gamma).fit(Xm_tr, ym_tr)
        rows.append([C, gamma, round(svm.score(Xm_tr, ym_tr), 3),
                     round(svm.score(Xm_te, ym_te), 3)])
print(pd.DataFrame(rows, columns=["C", "gamma", "Train acc", "Test acc"]))
