import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model.pkl")
FEATURES = json.load(open("features.json"))

# Valid ranges (taken from the training data) used to validate the form
FIELDS = [
    # name,                    label,                               min,  max,  step
    ("study_hours_per_day",    "Study hours per day",               0.5,  10,   0.1),
    ("sleep_hours",            "Sleep hours per night",             3,    10,   0.1),
    ("phone_usage_hours",      "Phone usage (hours per day)",       0.5,  12,   0.1),
    ("stress_level",           "Stress level (1 = low, 10 = high)", 1,    10,   1),
    ("focus_score",            "Focus score",                       30,   99,   1),
    ("attendance_percentage",  "Attendance (%)",                    40,   100,  0.1),
]


def interpret(score):
    if score >= 70:
        return "High productivity"
    if score >= 45:
        return "Medium productivity"
    return "Low productivity"


@app.route("/")
def home():
    return render_template("index.html", fields=FIELDS, values={}, prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    values = {name: request.form.get(name, "").strip() for name, *_ in FIELDS}
    try:
        row = {}
        for name, label, lo, hi, _ in FIELDS:
            v = float(values[name])
            if not lo <= v <= hi:
                raise ValueError(f"{label} must be between {lo} and {hi}.")
            row[name] = v
    except ValueError as e:
        msg = str(e) if "must be between" in str(e) else "Please fill every field with a valid number."
        return render_template("index.html", fields=FIELDS, values=values, prediction=None, error=msg), 400

    df = pd.DataFrame([row], columns=FEATURES)      # same column names/order as training
    score = float(min(max(model.predict(df)[0], 0), 100))
    return render_template("index.html", fields=FIELDS, values=values,
                           prediction=round(score, 2), label=interpret(score), error=None)


if __name__ == "__main__":
    app.run(debug=True)
