import numpy as np
import xgboost as xgb
import yaml
import os

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    xgb_params = params.get("xgboost", {})

    data = np.load("data/processed_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]

    model = xgb.XGBRegressor(
        max_depth=xgb_params.get("max_depth", 6),
        n_estimators=xgb_params.get("n_estimators", 100),
        learning_rate=xgb_params.get("learning_rate", 0.1),
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    # 使用 XGBoost 自带保存方法，保存为 JSON
    model.save_model("models/train_xgboost_model.json")

if __name__ == "__main__":
    main()
