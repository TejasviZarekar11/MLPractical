# 2 Make use of dimensionality reduction process using PCA and LDA on a suitable open-access high-dimensional dataset and observe the influence of feature reduction on model performance and computational efficiency.
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
X, y = load_iris(return_X_y=True)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Original Model
model = LogisticRegression(max_iter=1000)

start = time.time()
model.fit(X_train, y_train)
pred = model.predict(X_test)
original_time = time.time() - start

original_acc = accuracy_score(y_test, pred)

# PCA
pca = PCA(n_components=2)
P_train = pca.fit_transform(X_train)
P_test = pca.transform(X_test)

pca_model = LogisticRegression(max_iter=1000)

start = time.time()
pca_model.fit(P_train, y_train)
pca_pred = pca_model.predict(P_test)
pca_time = time.time() - start

pca_acc = accuracy_score(y_test, pca_pred)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
L_train = lda.fit_transform(X_train, y_train)
L_test = lda.transform(X_test)

lda_model = LogisticRegression(max_iter=1000)

start = time.time()
lda_model.fit(L_train, y_train)
lda_pred = lda_model.predict(L_test)
lda_time = time.time() - start

lda_acc = accuracy_score(y_test, lda_pred)

# Compare Results
results = pd.DataFrame({
    "Method": ["Original", "PCA", "LDA"],
    "Accuracy": [
        original_acc * 100,
        pca_acc * 100,
        lda_acc * 100
    ],
    "Execution Time": [
        original_time, pca_time, lda_time
    ]
})

print(results)

# Accuracy Comparison Graph
plt.bar(results["Method"], results["Accuracy"])

plt.xlabel("Method")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison: Original vs PCA vs LDA")

plt.show()

# PCA Visualization
plt.scatter(P_train[:, 0], P_train[:, 1], c=y_train)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

plt.show()

# LDA Visualization
plt.scatter(L_train[:, 0], L_train[:, 1], c=y_train)

plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.title("LDA Visualization")

plt.show()
