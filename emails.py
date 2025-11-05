# Engineered By Afan Shaikh Git: github.com/afan4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
# Load the dataset
df = pd.read_csv('emails.csv')
df.info()
print("Nulls in Last column:", df.iloc[:,-1].isnull().sum())
df = df.drop(df.columns[0], axis=1)
df=df.astype(int)
print(f"Unique values in Prediction: {df['Prediction'].unique()}")

# Split
X = df.drop('Prediction', axis=1)
y = df['Prediction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Split the data (stratify to handle class imbalance if any)

# K-Nearest Neighbors
print("\n=== K-Nearest Neighbors ===")
start = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_time = time.time() - start
knn_acc= accuracy_score(y_test, y_pred_knn)

print(f"Training time: {knn_time}s")
print(f"Accuracy: {knn_acc}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))


# Support Vector Machine
print("\n=== Support Vector Machine ===")
start = time.time()
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_time = time.time() - start
svm_acc= accuracy_score(y_test, y_pred_svm)

print(f"Training time: {svm_time}s")
print(f"Accuracy: {svm_acc}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))