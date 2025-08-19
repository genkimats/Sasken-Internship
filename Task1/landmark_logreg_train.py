# train_model_logistic.py
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load CSV
csv_file = "ear_dataset.csv"
print(f"[INFO] Loading dataset from {csv_file} ...")
data = pd.read_csv(csv_file)

X = data[["ear"]].values
y = data["label"].values

print(f"[INFO] Loaded {len(X)} samples.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (Logistic Regression)
print("[INFO] Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("[INFO] Evaluation Report:")
print(classification_report(y_test, y_pred))

# Save model
with open("models/drowsiness_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Model trained and saved as drowsiness_model.pkl")
