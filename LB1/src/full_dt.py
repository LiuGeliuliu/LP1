import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def main():
    data = np.load("data/processed_full.npz")
    X, y = data["X"], data["y"]

    model = joblib.load("models/train_dt_model.joblib")
    y_pred = model.predict(X)

    metrics = {
        "MSE": mean_squared_error(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/full_dt_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 保存特征重要性
    fi = model.feature_importances_
    pd.DataFrame({"feature_importance": fi}).to_csv("metrics/full_dt_feature_importance.csv", index=False)

if __name__ == "__main__":
    main()
