# 2 Make use of dimensionality reduction process using PCA and LDA on a suitable open-access high-dimensional dataset and observe the influence of feature reduction on model performance and computational efficiency.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

Cell 2: Load Dataset

```
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])
print("Target Classes:", data.target_names)
```

Cell 3: Split Dataset

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
```

Cell 4: Feature Scaling

```
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature Scaling Completed")
```

Cell 5: Train Model Using Original Features

```
model = LogisticRegression(max_iter=1000)

start = time.time()

model.fit(X_train, y_train)

original_time = time.time() - start

y_pred = model.predict(X_test)

original_accuracy = accuracy_score(y_test, y_pred)

print("Original Accuracy:", original_accuracy)
print("Original Training Time:", original_time)
```

Cell 6: Apply PCA

```
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA Training Shape:", X_train_pca.shape)
print("PCA Testing Shape:", X_test_pca.shape)

print("Explained Variance:", pca.explained_variance_ratio_)
print("Total Explained Variance:", sum(pca.explained_variance_ratio_))
```

Cell 7: Train Model Using PCA Features

```
pca_model = LogisticRegression(max_iter=1000)

start = time.time()

pca_model.fit(X_train_pca, y_train)

pca_time = time.time() - start

pca_pred = pca_model.predict(X_test_pca)

pca_accuracy = accuracy_score(y_test, pca_pred)

print("PCA Accuracy:", pca_accuracy)
print("PCA Training Time:", pca_time)
```

Cell 8: Apply LDA

```
lda = LinearDiscriminantAnalysis(n_components=1)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print("LDA Training Shape:", X_train_lda.shape)
print("LDA Testing Shape:", X_test_lda.shape)
```

Cell 9: Train Model Using LDA Features

```
lda_model = LogisticRegression(max_iter=1000)

start = time.time()

lda_model.fit(X_train_lda, y_train)

lda_time = time.time() - start

lda_pred = lda_model.predict(X_test_lda)

lda_accuracy = accuracy_score(y_test, lda_pred)

print("LDA Accuracy:", lda_accuracy)
print("LDA Training Time:", lda_time)
```

Cell 10: Compare Model Performance

```
results = pd.DataFrame({
    'Method': ['Original', 'PCA', 'LDA'],
    'Features': [30, 2, 1],
    'Accuracy': [
        original_accuracy,
        pca_accuracy,
        lda_accuracy
    ],
    'Training Time': [
        original_time,
        pca_time,
        lda_time
    ]
})

print(results)
```

Cell 11: Visualize Comparison

```
plt.bar(results['Method'], results['Accuracy'])

plt.title("Accuracy Comparison")
plt.xlabel("Method")
plt.ylabel("Accuracy")
plt.show()
```
