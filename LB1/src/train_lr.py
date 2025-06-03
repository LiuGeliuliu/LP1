import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

def main():
    data = np.load("data/processed_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/train_lr_model.joblib")

if __name__ == "__main__":
    main()
