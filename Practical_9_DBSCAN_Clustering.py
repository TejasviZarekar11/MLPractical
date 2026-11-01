"""
Practical 9: DBSCAN - Density-Based Clustering

Aim    : Apply DBSCAN to identify cluster structure and noise points.
Dataset: make_moons (non-spherical clusters) and Iris dataset (sklearn)

Run    : python Practical_9_DBSCAN_Clustering.py
"""

# ======================================================================
# Step 1
# ======================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Two moon shaped clusters + a few random outliers
X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
rng = np.random.RandomState(1)
X = np.vstack([X, rng.uniform(-1.5, 2.5, size=(10, 2))])
X = StandardScaler().fit_transform(X)
print("Data shape:", X.shape)

# ======================================================================
# Step 2
# ======================================================================
# Choose eps using the k-distance graph (k = min_samples)
min_samples = 5
dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
k_dist = np.sort(dist[:, -1])

plt.figure(figsize=(7, 3.5))
plt.plot(k_dist)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"Distance to {min_samples}th neighbour")
plt.title("k-distance graph (look for the elbow -> eps)")
plt.grid(True)
plt.show()

# ======================================================================
# Step 3
# ======================================================================
# Apply DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_                       # -1 means noise

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Clusters found :", n_clusters)
print("Noise points   :", list(labels).count(-1))
print("Core samples   :", len(db.core_sample_indices_))
print("Cluster sizes  :", {int(l): int((labels == l).sum()) for l in set(labels)})

# ======================================================================
# Step 4
# ======================================================================
# DBSCAN vs K-Means on the same data
km_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=km_labels, cmap="coolwarm", s=20)
ax[0].set_title("K-Means (k=2)")

noise = labels == -1
ax[1].scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise], cmap="viridis", s=20)
ax[1].scatter(X[noise, 0], X[noise, 1], c="red", marker="x", s=50, label="Noise")
ax[1].set_title("DBSCAN (eps=0.3, min_samples=5)")
ax[1].legend()
plt.tight_layout()
plt.show()

# ======================================================================
# Step 5
# ======================================================================
# DBSCAN on a real dataset (Iris)
Xi = StandardScaler().fit_transform(load_iris().data)
for eps in [0.4, 0.6, 0.8, 1.0]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(Xi)
    n = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps = {eps}: clusters = {n}, noise points = {list(lab).count(-1)}")
