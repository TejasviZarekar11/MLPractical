"""
Practical 2: Dimensionality Reduction using PCA and LDA

Aim    : Apply PCA and LDA on a high-dimensional dataset and observe the effect of feature reduction on model performance and computational efficiency.
Dataset: Digits dataset (sklearn, 1797 images, 64 features, 10 classes)

Run    : python Practical_2_PCA_LDA_Dimensionality_Reduction.py
"""

# ======================================================================
# Step 1
# ======================================================================
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)      # 8x8 pixel images -> 64 features
print("Dataset shape:", X.shape, "| Classes:", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ======================================================================
# Step 2
# ======================================================================
# PCA: how many components keep 95% of the variance?
pca_full = PCA().fit(X_train)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.argmax(cum_var >= 0.95) + 1
print("Components needed for 95% variance:", n95)

plt.figure(figsize=(7, 3.5))
plt.plot(range(1, 65), cum_var, marker="o", markersize=3)
plt.axhline(0.95, color="r", linestyle="--")
plt.axvline(n95, color="g", linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA - Explained Variance")
plt.grid(True)
plt.show()

# ======================================================================
# Step 3
# ======================================================================
# Compare model performance and training time
def evaluate(name, Xtr, Xte):
    times = []
    for _ in range(5):                          # repeat 5 times, keep the fastest
        start = time.perf_counter()
        model = SVC(kernel="rbf").fit(Xtr, y_train)
        times.append(time.perf_counter() - start)
    acc = accuracy_score(y_test, model.predict(Xte))
    return [name, Xtr.shape[1], round(acc, 4), round(min(times) * 1000, 1)]

results = [evaluate("Original (no reduction)", X_train, X_test)]

pca = PCA(n_components=n95).fit(X_train)
results.append(evaluate(f"PCA ({n95} components)",
                        pca.transform(X_train), pca.transform(X_test)))

pca10 = PCA(n_components=10).fit(X_train)
results.append(evaluate("PCA (10 components)",
                        pca10.transform(X_train), pca10.transform(X_test)))

lda = LDA(n_components=9).fit(X_train, y_train)    # max = classes - 1 = 9
results.append(evaluate("LDA (9 components)",
                        lda.transform(X_train), lda.transform(X_test)))

print(pd.DataFrame(results, columns=["Method", "Features", "Accuracy", "Fit time (ms)"]))

# ======================================================================
# Step 4
# ======================================================================
# Visualise the data in 2-D: PCA (unsupervised) vs LDA (supervised)
X_pca = PCA(n_components=2).fit_transform(X_train)
X_lda = LDA(n_components=2).fit_transform(X_train, y_train)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, data, title in zip(ax, [X_pca, X_lda], ["PCA - 2 components", "LDA - 2 components"]):
    sc_ = a.scatter(data[:, 0], data[:, 1], c=y_train, cmap="tab10", s=8)
    a.set_title(title)
fig.colorbar(sc_, ax=ax, label="Digit")
plt.show()
