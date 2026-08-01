import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =========================
# 1. LOAD DATASET
# =========================
data = pd.read_csv("../dataset/loan_approval_dataset.csv")

# Drop ID column
data.drop("loan_id", axis=1, inplace=True)

# =========================
# 2. HANDLE MISSING VALUES
# =========================
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# =========================
# 3. ENCODE CATEGORICAL DATA
# =========================
encoder = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = encoder.fit_transform(data[col])

# =========================
# 4. SPLIT FEATURES & TARGET
# =========================
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 6. TRAIN MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0
model_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_accuracies[name] = round(acc * 100, 2)

    print(f"{name} Accuracy: {acc:.2f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# =========================
# 7. SAVE BEST MODEL & SCALER
# =========================
with open("loan_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =========================
# 8. SAVE METRICS (IMPORTANT)
# =========================
with open("model_metrics.json", "w") as f:
    json.dump(model_accuracies, f, indent=4)

print("\nBest Model Saved Successfully!")
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("Metrics Saved to model_metrics.json")
