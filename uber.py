# Engineered By Afan Shaikh Git: github.com/afan4

# pip install pandas matplotlib scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('uber.csv')

# Basic cleaning
df = df.dropna(subset=['fare_amount', 'pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])
df = df[df['fare_amount'] > 0]

# Features
features = ['pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
X = df[features]
y = df['fare_amount']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Outliers using your exact syntax ---
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['fare_amount'] < lower_bound) | (df['fare_amount'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")

# --- Box plot ---
plt.figure(figsize=(6, 4))
df.boxplot(column='fare_amount')
plt.title('Box Plot of Fare Amount')
plt.show()

# --- Correlation matrix ---
corr_matrix = df[features + ['fare_amount']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(4))

# --- Models ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# --- Evaluation with simple print statements ---
r2_lr = r2_score(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
print(f"\nLinear Regression -> R²: {r2_lr:.4f}, RMSE: {rmse_lr:.4f}")

r2_rf = r2_score(y_test, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
print(f"Random Forest     -> R²: {r2_rf:.4f}, RMSE: {rmse_rf:.4f}")