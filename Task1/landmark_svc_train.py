# train_model_from_csv.py
import pandas as pd
import pickle
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load CSV
csv_file = "ear_labels.csv"
print(f"[INFO] Loading dataset from {csv_file} ...")
data = pd.read_csv(csv_file)

# Extract filename and person ID
data["filename"] = data["filepath"].apply(lambda x: x.split("/")[-1])
data["person_id"] = data["filename"].str[0]

# Extract subdirectory (second part of the path)
data["subdir"] = data["filepath"].apply(lambda x: os.path.normpath(x).split(os.sep)[1])

# Create label: 1 = drowsy, 0 = non_drowsy
data["label"] = data["subdir"].apply(lambda x: 1 if x == "drowsy" else 0)

# Shuffle before grouping
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit to 100 samples per person
data = (
    data.groupby("person_id", group_keys=False)
        .apply(lambda g: g.head(100))
        .reset_index(drop=True)
)

# Show class distribution
print("[INFO] Class distribution after filtering:")
print(data["label"].value_counts())

# Prepare features and labels
X = data[["ear"]].values
y = data["label"].values

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVC model
print("[INFO] Training SVC model...")
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("[INFO] Evaluation Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
with open("models/svc_ear.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Model trained and saved as models/svc_ear.pkl")
