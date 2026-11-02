# 9 Implement an ensemble learning technique such as Random Forest or Boosting and compare its performance with a baseline model.
# Import required libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
```

Explanation:

* `load_breast_cancer` loads the dataset.
* `DecisionTreeClassifier` creates the baseline model.
* `RandomForestClassifier` creates the Random Forest model.
* `AdaBoostClassifier` creates the Boosting model.
* Evaluation functions calculate model performance.

Cell 2: Load Dataset

```
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)
```

Explanation:

* `X` contains the 30 input features.
* `y` contains the target classes.
* The dataset has 569 records and 30 features.

Cell 3: Split Dataset

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
```

Explanation:

* 80% of data is used for training.
* 20% of data is used for testing.
* `random_state=42` helps produce a reproducible split.

Cell 4: Train Baseline Decision Tree

```
dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
```

Explanation:

* Creates a Decision Tree model.
* `fit()` trains the model.
* `predict()` predicts the test data.

Cell 5: Train Random Forest

```
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
```

Explanation:

* `n_estimators=100` means 100 decision trees are created.
* Each tree makes predictions.
* Random Forest combines their predictions using majority voting.

Cell 6: Train AdaBoost

```
ab = AdaBoostClassifier(random_state=42)

ab.fit(X_train, y_train)

y_pred_ab = ab.predict(X_test)
```

Explanation:

* Creates an AdaBoost model.
* It trains weak learners sequentially.
* It gives more importance to incorrectly classified records.

Cell 7: Calculate Performance Metrics

```
models = {
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "AdaBoost": y_pred_ab
}

results = []

for name, pred in models.items():

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([
        name, accuracy, precision, recall, f1
    ])

df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print(df)
```

Explanation:

* Calculates four evaluation metrics.
* Stores the results of all three models.
* Displays them in a comparison table.

Cell 8: Visualize Model Performance

```
plt.figure(figsize=(8, 5))

plt.bar(df["Model"], df["Accuracy"])

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Ensemble Learning Performance Comparison")

plt.show()
```
