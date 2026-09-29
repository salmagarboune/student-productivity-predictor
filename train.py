"""Re-creates model.pkl and features.json with YOUR installed scikit-learn version.
Run: python train.py   (same model as the notebook: Linear Regression on the 6 features)"""
import json, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

FEATURES = ["study_hours_per_day", "sleep_hours", "phone_usage_hours",
            "stress_level", "focus_score", "attendance_percentage"]
df = pd.read_csv("data/train.csv")
model = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("sc", StandardScaler()),
                  ("model", LinearRegression())]).fit(df[FEATURES], df["productivity_score"])
joblib.dump(model, "model.pkl")
json.dump(FEATURES, open("features.json", "w"))
print("model.pkl saved, R2 on train =", round(model.score(df[FEATURES], df["productivity_score"]), 5))
