# 7 Utilize Support Vector Machine classifiers with linear and nonlinear kernels on a suitable open-access dataset and compare classification performance using appropriate evaluation metrics.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM models
linear_model = SVC(kernel='linear')
rbf_model = SVC(kernel='rbf')

# Train models
linear_model.fit(X_train, y_train)
rbf_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_rbf = rbf_model.predict(X_test)

# Calculate accuracy
linear_accuracy = accuracy_score(y_test, y_pred_linear)
rbf_accuracy = accuracy_score(y_test, y_pred_rbf)

print("Linear SVM Accuracy:", linear_accuracy)
print("RBF SVM Accuracy:", rbf_accuracy)

# Classification reports
print("\nLinear SVM Report:")
print(classification_report(y_test, y_pred_linear))

print("\nRBF SVM Report:")
print(classification_report(y_test, y_pred_rbf))

# Confusion matrices
cm_linear = confusion_matrix(y_test, y_pred_linear)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues')
plt.title("Linear SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Greens')
plt.title("RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# Accuracy comparison
results = pd.DataFrame({
    "SVM Kernel": ["Linear", "RBF"],
    "Accuracy": [linear_accuracy, rbf_accuracy]
})

print("\nAccuracy Comparison:")
print(results)

sns.barplot(x="SVM Kernel", y="Accuracy", data=results)
plt.title("Linear vs Nonlinear SVM")
plt.ylim(0, 1)
plt.show()
