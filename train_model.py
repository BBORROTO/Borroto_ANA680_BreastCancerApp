# train_model.py

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle

# 1. Fetch dataset
data = fetch_ucirepo(id=15)
X_raw = data.data.features
y_raw = data.data.targets

# 2. Drop missing values
X_clean = X_raw.dropna()
y_clean = y_raw.loc[X_clean.index]

# 3. Encode target: 2 → 0 (Benign), 4 → 1 (Malignant)
y_encoded = y_clean["Class"].map({2: 0, 4: 1})

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# 7. Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ XGBoost model and scaler saved as model.pkl and scaler.pkl")