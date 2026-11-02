# 9 Employ a density-based clustering algorithm such as DBSCAN using a suitable open-access dataset to identify the clustering structure and noise points.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
```

Cell 2: Generate Dataset

```
X, y = make_moons(
    n_samples=300,
    noise=0.08,
    random_state=42
)

print("Dataset Shape:", X.shape)
```

Explanation:

* `n_samples=300` creates 300 data points.
* `noise=0.08` adds some random variation.
* `random_state=42` ensures reproducible data.

Cell 3: Display Original Dataset

```
plt.scatter(X[:, 0], X[:, 1])

plt.title("Original Dataset")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
```

Explanation: This graph displays the original data points before clustering.
Cell 4: Feature Scaling

```
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Data Scaling Completed")
```

Explanation: StandardScaler brings both features to a common scale because DBSCAN uses distance calculations.
Cell 5: Apply DBSCAN

```
dbscan = DBSCAN(
    eps=0.25,
    min_samples=5
)

labels = dbscan.fit_predict(X_scaled)

print("Cluster Labels:")
print(labels)
```

Important lines:

* `eps=0.25` specifies the neighborhood distance.
* `min_samples=5` specifies the minimum number of points required to form a dense region.
* `fit_predict()` performs clustering and returns cluster labels.

Cell 6: Count Clusters and Noise Points

```
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

n_noise = np.sum(labels == -1)

print("Number of Clusters:", n_clusters)
print("Number of Noise Points:", n_noise)
```

Explanation:

* `n_clusters` counts the clusters while excluding noise.
* `n_noise` counts all points labelled `-1`.

Cell 7: Identify Core Points

```
core_points = dbscan.core_sample_indices_

print("Number of Core Points:", len(core_points))
```

Explanation: `core_sample_indices_` returns the indices of core points identified by DBSCAN.
Cell 8: Visualize Clusters and Noise Points

```
core_mask = np.zeros(len(labels), dtype=bool)

core_mask[core_points] = True

plt.figure(figsize=(8, 5))

for k in set(labels):

    if k == -1:
        color = 'black'
    else:
        color = plt.cm.Spectral(k / max(n_clusters, 1))

    # Core points
    plt.scatter(
        X_scaled[(labels == k) & core_mask, 0],
        X_scaled[(labels == k) & core_mask, 1],
        c=[color],
        marker='o',
        label=f'Cluster {k} Core'
    )

    # Border points and noise
    if k == -1:
        plt.scatter(
            X_scaled[labels == k, 0],
            X_scaled[labels == k, 1],
            c='black',
            marker='x',
            label='Noise'
        )
    else:
        plt.scatter(
            X_scaled[(labels == k) & (~core_mask), 0],
            X_scaled[(labels == k) & (~core_mask), 1],
            c=[color],
            marker='x',
            label=f'Cluster {k} Border'
        )

plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.show()
```
