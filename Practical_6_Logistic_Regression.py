# 6 Illustrate the performance of a binary classification model using logistic regression on a suitable open-access dataset through a confusion matrix, precision, recall, and F1-score.

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
```

Cell 2: Load Dataset

```
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)
```

Explanation: `load_breast_cancer()` loads the built-in dataset. `X` contains input features and `y` contains target classes.
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

Explanation: StandardScaler converts features into a common scale, helping Logistic Regression perform effectively.
Cell 5: Create and Train Model

```
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Training Completed")
```

Cell 6: Make Predictions

```
y_pred = model.predict(X_test)

print("Actual Values:", y_test[:10])
print("Predicted Values:", y_pred[:10])
```

Cell 7: Confusion Matrix

```
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

Cell 8: Visualize Confusion Matrix

```
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")

plt.show()
```

Explanation: The heatmap displays correct and incorrect predictions. The diagonal values represent correct predictions.
Cell 9: Calculate Precision, Recall and F1-Score

```
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
```

Cell 10: Classification Report

```
print(classification_report(
    y_test, y_pred,
    target_names=data.target_names
))
```
