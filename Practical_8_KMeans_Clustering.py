# 8 Apply the K-Means clustering algorithm on a suitable open-access dataset to obtain the optimal number of clusters using the Elbow or Silhouette method
Cell 1: Import Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

Cell 2: Load Dataset

```
data = load_iris()

X = data.data

df = pd.DataFrame(X, columns=data.feature_names)

print("Dataset Shape:", df.shape)
print(df.head())
```

Explanation: `load_iris()` loads the Iris dataset. We use its four numerical features for clustering.
Cell 3: Feature Scaling

```
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Data Scaling Completed")
```

Explanation: StandardScaler brings all features to a common scale so that features with larger values do not dominate distance calculations.
Cell 4: Calculate WCSS

```
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

print(wcss)
```

Important lines:

* `range(1, 11)` tests cluster numbers from 1 to 10.
* `n_clusters=i` specifies the number of clusters.
* `inertia_` returns the WCSS value.

Cell 5: Plot Elbow Graph

```
plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.show()
```

Explanation: Observe the graph and identify the point where the WCSS reduction starts slowing down. For the Iris dataset, K = 3 is a commonly observed suitable choice.
Cell 6: Apply K-Means Clustering

```
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

y_kmeans = kmeans.fit_predict(X_scaled)

print("Cluster Labels:")
print(y_kmeans)
```

Explanation: Based on the Elbow Method, we apply K-Means with three clusters.
Cell 7: Add Cluster Labels to Dataset

```
df['Cluster'] = y_kmeans

print(df.head(10))
```

Cell 8: Visualize Clusters

```
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=y_kmeans,
    cmap='viridis'
)

plt.xlabel("Sepal Length (Scaled)")
plt.ylabel("Sepal Width (Scaled)")
plt.title("K-Means Clustering")

plt.show()
```







