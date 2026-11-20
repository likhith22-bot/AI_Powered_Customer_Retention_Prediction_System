import pickle

import numpy as np
from flask import Flask, render_template, request

with open("scaledown.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Model.pkl", "rb") as f:
    reg = pickle.load(f)

app = Flask(__name__)

# Exact column order the scaler and model were trained on
COLUMNS = [
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'Contract_res', 'SeniorCitizen', 'tenure', 'MonthlyCharges',
    'TotalCharges_mod',
]

# Change this if your ordinal encoder used a different order
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]


def build_row(f):
    """Turn the raw form values into the encoded feature vector."""
    phone = f.get("PhoneService", "Yes")
    internet = f.get("InternetService", "DSL")

    row = {
        "gender_Male": f.get("gender") == "Male",
        "Partner_Yes": f.get("Partner") == "Yes",
        "Dependents_Yes": f.get("Dependents") == "Yes",
        "PhoneService_Yes": phone == "Yes",
        "InternetService_Fiber optic": internet == "Fiber optic",
        "InternetService_No": internet == "No",
        "PaperlessBilling_Yes": f.get("PaperlessBilling") == "Yes",
        "Contract_res": CONTRACT_MAP[f.get("Contract", "Month-to-month")],
        "SeniorCitizen": int(f.get("SeniorCitizen", 0)),
        "tenure": float(f["tenure"]),
        "MonthlyCharges": float(f["MonthlyCharges"]),
        "TotalCharges_mod": float(f["TotalCharges"]),
    }

    lines = "No phone service" if phone == "No" else f.get("MultipleLines", "No")
    row["MultipleLines_No phone service"] = lines == "No phone service"
    row["MultipleLines_Yes"] = lines == "Yes"

    for name in ADDONS:
        value = "No internet service" if internet == "No" else f.get(name, "No")
        row[f"{name}_No internet service"] = value == "No internet service"
        row[f"{name}_Yes"] = value == "Yes"

    pay = f.get("PaymentMethod", "")
    row["PaymentMethod_Credit card (automatic)"] = pay == "Credit card (automatic)"
    row["PaymentMethod_Electronic check"] = pay == "Electronic check"
    row["PaymentMethod_Mailed check"] = pay == "Mailed check"
    # "Bank transfer (automatic)" is the dropped baseline, so all three stay 0

    return [float(row[c]) for c in COLUMNS]


@app.route("/")
def main_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_fun():
    form = request.form.to_dict()
    try:
        val = np.array([build_row(form)])
    except (ValueError, KeyError):
        return render_template(
            "index.html", form=form,
            error="Check the numeric fields: tenure and charges must be valid numbers.",
        )

    val1 = scaler.transform(val)

    probability = None
    if hasattr(reg, "predict_proba"):
        probability = float(reg.predict_proba(val1)[0][1])
        is_churn = probability >= 0.5
    else:
        is_churn = float(reg.predict(val1)[0]) >= 0.5

    return render_template(
        "index.html",
        prediction_text="Likely to churn" if is_churn else "Likely to stay",
        is_churn=is_churn,
        probability=probability,
        form=form,
    )


if __name__ == "__main__":
    app.run(debug=True)