import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

# Create graphs folder
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to existing static/graphs folder
GRAPH_DIR = os.path.join(BASE_DIR, "static", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


# Load dataset
data = pd.read_csv("../dataset/loan_approval_dataset.csv")
data.drop("loan_id", axis=1, inplace=True)

# Handle missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Encode categorical features
encoder = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = encoder.fit_transform(data[col])

X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# Confusion Matrix
cm = confusion_matrix(y_test, rf.predict(X_test))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
plt.close()

# Accuracy comparison
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": rf,
    "SVM": SVC()
}

accuracies = []
for model in models.values():
    model.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(7, 4))
plt.bar(models.keys(), accuracies)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Models")
plt.savefig(os.path.join(GRAPH_DIR, "accuracy.png"))
plt.close()

print("Graphs generated successfully!")
