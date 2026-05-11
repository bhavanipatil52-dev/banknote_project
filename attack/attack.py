import numpy as np
import requests
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

API_URL = "http://localhost:5000/predict"
N_QUERIES = 1000

# Feature ranges based on banknote dataset
FEATURE_RANGES = {
    "variance":  (-7.0,  7.0),
    "skewness":  (-14.0, 14.0),
    "curtosis":  (-5.0,  18.0),
    "entropy":   (-8.5,  2.5),
}

rng = np.random.default_rng(42)

print(f"Sending {N_QUERIES} queries to the API...")

rows = []
for i in range(N_QUERIES):
    features = [
        float(rng.uniform(lo, hi))
        for lo, hi in FEATURE_RANGES.values()
    ]
    try:
        response = requests.post(API_URL, json={"features": features}, timeout=5)
        prediction = response.json()["prediction"]
        rows.append(features + [prediction])
    except Exception as e:
        print(f"Query {i+1} failed: {e}")

    if (i + 1) % 200 == 0:
        print(f"{i+1} queries done, collected {len(rows)} samples")

# Save stolen data
df = pd.DataFrame(rows, columns=["variance", "skewness", "curtosis", "entropy", "label"])
df.to_csv("stolen_data.csv", index=False)
print(f"Stolen data saved: {len(rows)} samples")

# Train copycat model
X = df[["variance", "skewness", "curtosis", "entropy"]].values
y = df["label"].values

copycat = RandomForestClassifier(n_estimators=100, random_state=42)
copycat.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(copycat, "model/copycat_model.pkl")
print("Copycat model saved")

# Evaluate agreement
print("\nEvaluating agreement on 300 fresh inputs...")
X_test = np.column_stack([
    rng.uniform(lo, hi, 300)
    for lo, hi in FEATURE_RANGES.values()
])

victim_preds = []
for row in X_test:
    try:
        r = requests.post(API_URL, json={"features": row.tolist()}, timeout=5)
        victim_preds.append(r.json()["prediction"])
    except:
        victim_preds.append(-1)

victim_preds = np.array(victim_preds)
valid = victim_preds != -1
copycat_preds = copycat.predict(X_test[valid])

agreement = accuracy_score(victim_preds[valid], copycat_preds)
print(f"Agreement Rate: {agreement * 100:.2f}%")
print(classification_report(victim_preds[valid], copycat_preds,
      target_names=["Real", "Fake"], zero_division=0))

if agreement >= 0.90:
    print("HIGH RISK: Attack was successful")
elif agreement >= 0.75:
    print("MEDIUM RISK: Partially successful")
else:
    print("LOW RISK: More queries needed")