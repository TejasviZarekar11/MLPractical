"""
Practical 10: Ensemble Learning - Random Forest and Boosting vs Baseline

Aim    : Analyse ensemble learning techniques (Random Forest, AdaBoost, Gradient Boosting) against a baseline Decision Tree.
Dataset: Wine dataset (sklearn, 178 samples, 13 features, 3 classes)

Run    : python Practical_10_Ensemble_Learning.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier)
from sklearn.metrics import accuracy_score, f1_score

data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print("Shape:", X.shape, "| Classes:", np.unique(y))

# ======================================================================
# Step 2
# ======================================================================
# Baseline model + ensemble models
models = {
    "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    cv = cross_val_score(m, X, y, cv=5).mean()
    rows.append([name, accuracy_score(y_test, pred),
                 f1_score(y_test, pred, average="macro"), cv])

result = pd.DataFrame(rows, columns=["Model", "Test Accuracy", "Test F1 (macro)",
                                     "5-fold CV Accuracy"]).round(4)
print(result)

# ======================================================================
# Step 3
# ======================================================================
# Compare visually
plt.figure(figsize=(8, 4))
plt.barh(result["Model"], result["5-fold CV Accuracy"], color="steelblue")
plt.xlim(0.8, 1.0)
plt.xlabel("5-fold cross-validation accuracy")
plt.title("Baseline vs Ensemble models")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ======================================================================
# Step 4
# ======================================================================
# Feature importance from the Random Forest
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values()

imp.tail(8).plot(kind="barh", figsize=(7, 4), color="darkgreen")
plt.xlabel("Importance")
plt.title("Top 8 features - Random Forest")
plt.tight_layout()
plt.show()
