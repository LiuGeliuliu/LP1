import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def main():
    data = np.load("data/processed_split.npz")
    X_val, y_val = data["X_val"], data["y_val"]

    model = joblib.load("models/train_catboost_model.joblib")
    y_pred = model.predict(X_val)

    metrics = {
        "MSE": mean_squared_error(y_val, y_pred),
        "MAE": mean_absolute_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluate_catboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    fi = model.get_feature_importance()
    pd.DataFrame({"feature_importance": fi}).to_csv("metrics/evaluate_catboost_feature_importance.csv", index=False)

if __name__ == "__main__":
    main()
