# 7 Utilize Support Vector Machine classifiers with linear and nonlinear kernels on a suitable open-access dataset and compare classification performance using appropriate evaluation metrics.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
```

Cell 2: Load Dataset

```
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)
```

Explanation: `load_breast_cancer()` loads the dataset. `X` contains input features and `y` contains target classes.
Cell 3: Split Dataset

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
```

Cell 4: Feature Scaling

```
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Important: Feature scaling is necessary for SVM because it uses distances and margins to find the decision boundary.
Cell 5: Create Linear SVM Model

```
linear_model = SVC(kernel='linear')

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

print("Linear SVM Training Completed")
```

Important line: `kernel='linear'` specifies the Linear kernel.
Cell 6: Create Nonlinear SVM Model

```
rbf_model = SVC(kernel='rbf')

rbf_model.fit(X_train, y_train)

rbf_pred = rbf_model.predict(X_test)

print("RBF SVM Training Completed")
```

Important line: `kernel='rbf'` specifies the nonlinear Radial Basis Function kernel.
Cell 7: Evaluate Both Models

```
results = []

models = {
    "Linear SVM": linear_pred,
    "RBF SVM": rbf_pred
}

for name, pred in models.items():

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([
        name, accuracy, precision, recall, f1
    ])

result_df = pd.DataFrame(
    results,
    columns=[
        "Model", "Accuracy", "Precision",
        "Recall", "F1-Score"
    ]
)

print(result_df)
```

Cell 8: Confusion Matrix for Linear SVM

```
cm1 = confusion_matrix(y_test, linear_pred)

sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')

plt.title("Linear SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
```

Cell 9: Confusion Matrix for RBF SVM

```
cm2 = confusion_matrix(y_test, rbf_pred)

sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens')

plt.title("RBF SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
```

Cell 10: Compare Model Performance

```
result_df.set_index("Model").plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Linear SVM vs RBF SVM")
plt.ylabel("Performance Score")
plt.ylim(0, 1.1)

plt.show()
```
