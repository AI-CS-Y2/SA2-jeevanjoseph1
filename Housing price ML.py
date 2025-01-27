# necessary libraries for the assessment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv("housing_price_dataset.csv")
print(data.head(10))

# Converting categorical 'Neighborhood' into numerical values
data = pd.get_dummies(data, columns=["Neighborhood"], drop_first=True)

# Preparing data for Linear Regression
X_linear = data.drop(["Price"], axis=1)
y_linear = data["Price"]

# Preparing data for Logistic Regression 
median_price = data["Price"].median()
data["Price_Class"] = (data["Price"] > median_price).astype(int)
X_logistic = data.drop(["Price", "Price_Class"], axis=1)
y_logistic = data["Price_Class"]

# Split data into training and testing sets
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_logistic, y_logistic, test_size=0.2, random_state=42)

#Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_lr, y_train_lr)
y_pred_linear = linear_model.predict(X_test_lr)

#Linear Regression Graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test_lr, y_pred_linear, alpha=0.7)
plt.plot([y_test_lr.min(), y_test_lr.max()], [y_test_lr.min(), y_test_lr.max()], 'r--')
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

#Logistic Regression
logistic_model = LogisticRegression(max_iter=3500, random_state=50)
logistic_model.fit(X_train_clf, y_train_clf)
y_pred_logistic = logistic_model.predict(X_test_clf)
y_proba_logistic = logistic_model.predict_proba(X_test_clf)[:, 1]

# Evaluataion metrics for Logistic Regression
print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_logistic):.2f}")
print(f"Precision: {precision_score(y_test_clf, y_pred_logistic):.2f}")
print(f"Recall: {recall_score(y_test_clf, y_pred_logistic):.2f}")
print(f"F1 Score: {f1_score(y_test_clf, y_pred_logistic):.2f}")
print(f"Log Loss: {log_loss(y_test_clf, y_proba_logistic):.2f}")
print(f"ROC AUC: {roc_auc_score(y_test_clf, y_proba_logistic):.2f}")

#Logistic Regression Graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test_clf, y_pred_logistic, alpha=0.7)
plt.plot([y_test_clf.min(), y_test_clf.max()], [y_test_clf.min(), y_test_clf.max()], 'r--')
plt.title("Logistic Regression: Actual vs Predicted")
plt.xlabel("Actual Classes")
plt.ylabel("Predicted Classes")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_clf, y_proba_logistic)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="Logistic Regression (AUC = {:.2f})".format(roc_auc_score(y_test_clf, y_proba_logistic)))
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve (Logistic Regression)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test_clf, y_proba_logistic)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="Logistic Regression")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
