from svm_rbf import SVM_RBF  
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("diabetes_cleaning.csv")
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SVM_RBF(C=1.0, gamma=0.05, lr=0.001, n_iters=100)
model.fit(X_scaled, y)

# Simpan model dan scaler
with open("model_svm.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
