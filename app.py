from flask import Flask, render_template, request, jsonify
import sqlite3
import pickle
import numpy as np
import json

import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


app = Flask(__name__)

prediction_history = []
# Load trained model
model = pickle.load(open("model/loan_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/models")
def models():
    with open("model/model_metrics.json", "r") as f:
        metrics = json.load(f)

    return render_template(
        "models.html",
        metrics=metrics
    )

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get input values from form
        no_of_dependents = int(request.form["dependents"])
        education = int(request.form["education"])
        self_employed = int(request.form["self_employed"])
        income_annum = float(request.form["applicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        cibil_score = int(request.form["credit_history"])
        residential_assets_value = float(request.form["residential_assets_value"])
        commercial_assets_value = float(request.form["commercial_assets_value"])
        luxury_assets_value = float(request.form["luxury_assets_value"])
        bank_asset_value = float(request.form["bank_asset_value"])

        # Arrange input in correct order
        input_data = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        # Scale input
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        result = "✅ Loan Approved" if prediction[0] == 0 else "❌ Loan Rejected"

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (result) VALUES (?)", (result,))
        conn.commit()
        conn.close()

        prediction_history.append(result)
        approved = prediction_history.count("Loan Approved")
        rejected = prediction_history.count("Loan Rejected")
        return render_template(
            'results.html',
            result=result,
            history=prediction_history,
            approved=approved,
            rejected=rejected
        )
    except Exception as e:
        return render_template("index.html", prediction_text = "Error occured!")
    
def generate_graphs():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    data = pd.read_csv("dataset/loan_approval_dataset.csv")
    data.drop("Loan_ID", axis=1, inplace=True)

    for col in data.columns:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)

    encoder = LabelEncoder()
    for col in data.select_dtypes(include="object").columns:
        data[col] = encoder.fit_transform(data[col])

    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    accuracies = []

    for model in models.values():
        model.fit(X_train, y_train)
        accuracies.append(accuracy_score(y_test, model.predict(X_test)))

    # Accuracy Graph
    plt.figure()
    plt.bar(models.keys(), accuracies)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.savefig("static/graphs/accuracy.png")
    plt.close()

    # Confusion Matrix (Random Forest)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    cm = confusion_matrix(y_test, rf.predict(X_test))

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("static/graphs/confusion_matrix.png")
    plt.close()
    

@app.route("/prediction-data")
def prediction_data():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE '%Approved%'")
    approved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE '%Rejected%'")
    rejected = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        "approved": approved,
        "rejected": rejected
    })


if __name__ == "__main__":
    app.run(debug=True)

