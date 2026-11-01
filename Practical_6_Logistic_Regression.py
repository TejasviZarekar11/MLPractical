"""
Practical 6: Logistic Regression - Binary Classification

Aim    : Illustrate the performance of a binary logistic regression classifier using the confusion matrix, precision, recall and F1-score.
Dataset: Breast Cancer Wisconsin dataset (sklearn, 569 samples, 30 features, malignant/benign)

Run    : python Practical_6_Logistic_Regression.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)

data = load_breast_cancer()
X, y = data.data, data.target
print("Shape:", X.shape)
print("Classes:", list(data.target_names), "-> counts", list(np.bincount(y)))

# ======================================================================
# Step 2
# ======================================================================
# Split, scale and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))

# ======================================================================
# Step 3
# ======================================================================
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# ======================================================================
# Step 4
# ======================================================================
# Precision, recall, F1-score
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))
print()
print(classification_report(y_test, y_pred, target_names=data.target_names))
