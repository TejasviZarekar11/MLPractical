"""
Practical 8: K-Means Clustering - Elbow and Silhouette Methods

Aim    : Apply K-Means clustering and find the optimal number of clusters using the Elbow and Silhouette methods.
Dataset: Iris dataset (sklearn, 150 flowers, 4 features; labels used only for checking)

Run    : python Practical_8_KMeans_Clustering.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)       # clustering uses distances -> scale
print("Data shape:", X.shape)

# ======================================================================
# Step 2
# ======================================================================
# Elbow method (inertia / WCSS) and Silhouette score for k = 2..10
k_range = range(2, 11)
inertia, sil = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

print(pd.DataFrame({"k": list(k_range), "Inertia": np.round(inertia, 1),
                    "Silhouette": np.round(sil, 3)}).to_string(index=False))

# ======================================================================
# Step 3
# ======================================================================
# Plot both curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(k_range, inertia, marker="o")
ax[0].set_xlabel("Number of clusters (k)"); ax[0].set_ylabel("Inertia (WCSS)")
ax[0].set_title("Elbow Method")
ax[1].plot(k_range, sil, marker="o", color="green")
ax[1].set_xlabel("Number of clusters (k)"); ax[1].set_ylabel("Silhouette score")
ax[1].set_title("Silhouette Method")
plt.tight_layout()
plt.show()

best_k = list(k_range)[int(np.argmax(sil))]
print("Best k by silhouette score:", best_k)

# ======================================================================
# Step 4
# ======================================================================
# Final model with k = 3 (3 species of iris) and visualisation
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
print("Silhouette score (k=3):", round(silhouette_score(X, km.labels_), 3))
print("Adjusted Rand Index vs true species:",
      round(adjusted_rand_score(iris.target, km.labels_), 3))
print("Cluster sizes:", np.bincount(km.labels_))

X2 = PCA(n_components=2).fit_transform(X)
centers = PCA(n_components=2).fit(X).transform(km.cluster_centers_)
plt.figure(figsize=(7, 4.5))
plt.scatter(X2[:, 0], X2[:, 1], c=km.labels_, cmap="viridis", s=30)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means Clusters (k=3) on PCA axes")
plt.legend()
plt.show()
