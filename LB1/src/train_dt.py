import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib
import yaml
import os

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    dt_params = params.get("dt", {})

    data = np.load("data/processed_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]

    model = DecisionTreeRegressor(
        max_depth=dt_params.get("max_depth"),
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/train_dt_model.joblib")

if __name__ == "__main__":
    main()
