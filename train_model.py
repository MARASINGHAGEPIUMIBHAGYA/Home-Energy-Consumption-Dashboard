import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/household_power_consumption.txt", sep=';', low_memory=False, na_values='?')
df = df[['Global_active_power']].dropna()
df['Global_active_power'] = df['Global_active_power'].astype(float)

# Use last 5000 points
series = df['Global_active_power'].values[-5000:].reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Create sequences (features and targets)
seq_length = 24
X, y = [], []
for i in range(len(series_scaled) - seq_length):
    # Flatten the sequence window for classical ML input
    X.append(series_scaled[i:i + seq_length].flatten())
    y.append(series_scaled[i + seq_length][0])

X = np.array(X)
y = np.array(y)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")
