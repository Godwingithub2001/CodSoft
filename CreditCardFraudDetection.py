import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv("C:\\DATASCIENCE\\MyProjects\\CodSoft\\dataset\\creditcard.csv")

print(df.describe())

print(df.isnull().sum())

x = df.drop('Class', axis=1)
y = df['Class']  # (0 = geniune, 1 = fraud)

# normalize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
# is used to oversample the minority class (fraudulent transactions) to balance the dataset.

smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_scaled, y)

# Split the data into training and testing

x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)


# Train Logistic Regression model

lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train, y_train)
lr_preds = lr_model.predict(x_test)
print(lr_preds)

# Evaluate Logistic Regression
# Precision: Measures the accuracy of the positive predictions (fraudulent transactions).
# Recall: Measures the ability of the model to identify all fraudulent transactions.
# F1-Score: Balances precision and recall, especially useful for imbalanced datasets.

print("\nLogistic Regression Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_preds))
print("\nClassification Report:")
print(classification_report(y_test, lr_preds))
print(f"Precision: {precision_score(y_test, lr_preds):.4f}")
print(f"Recall: {recall_score(y_test, lr_preds):.4f}")
print(f"F1-Score: {f1_score(y_test, lr_preds):.4f}")




