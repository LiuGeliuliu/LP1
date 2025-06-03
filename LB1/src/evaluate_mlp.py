import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    data = np.load("data/processed_split.npz")
    X_val = data["X_val"]  # numpy array
    y_val = data["y_val"]

    model = joblib.load("models/train_mlp_model.joblib")

    y_pred = model.predict(X_val)

    metrics = {
        "MSE": mean_squared_error(y_val, y_pred),
        "MAE": mean_absolute_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluate_mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 如果有特征重要性，保存对应csv，MLP无此项，若有请补充代码

if __name__ == "__main__":
    main()
