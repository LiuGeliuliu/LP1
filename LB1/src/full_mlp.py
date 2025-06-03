# src/full_mlp.py
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter

def main():
    # 1. 加载全量数据
    data = np.load("data/processed_full.npz")
    X_full = data["X"]
    y_full = data["y"]

    # 2. 加载训练好的模型
    model = joblib.load("models/train_mlp_model.joblib")

    # 3. 预测
    y_pred = model.predict(X_full)

    # 4. 计算指标
    mse = mean_squared_error(y_full, y_pred)
    mae = mean_absolute_error(y_full, y_pred)
    r2 = r2_score(y_full, y_pred)

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }

    # 5. 保存指标到json文件
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/full_mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. 写TensorBoard日志（训练过程无，写预测指标和权重直方图）
    writer = SummaryWriter(log_dir="runs/full_mlp")
    writer.add_scalar("Metrics/MSE", mse, 0)
    writer.add_scalar("Metrics/MAE", mae, 0)
    writer.add_scalar("Metrics/R2", r2, 0)

    # 写权重分布直方图（第0步）
    for i, coef in enumerate(model.coefs_):
        writer.add_histogram(f"Weights/layer_{i}", coef, 0)

    writer.close()

    print(f"Full dataset evaluation completed.")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()