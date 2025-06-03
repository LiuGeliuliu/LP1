import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

def main():
    # 1. 读取参数
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    mlp_params = params.get("mlp", {})

    # 2. 加载训练数据
    data = np.load("data/processed_split.npz")
    X = data["X_train"]
    y = data["y_train"]

    max_iter = mlp_params.get("max_iter", 100)
    hidden_layer_sizes = tuple(mlp_params.get("hidden_layer_sizes", (100,)))
    activation = mlp_params.get("activation", "relu")
    solver = mlp_params.get("solver", "adam")

    # 3. 初始化模型（warm_start=True 以支持逐轮训练）
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=1,       # 每次仅迭代 1 次，配合 warm_start
        warm_start=True,
        random_state=42
    )

    # 4. 初始化 TensorBoard 写入器
    writer = SummaryWriter(log_dir="runs/train_mlp")

    # 5. 循环训练并记录指标 & 权重直方图
    for epoch in range(max_iter):
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        writer.add_scalar("train/mse", mse, epoch)
        writer.add_scalar("train/loss", model.loss_, epoch)

        # ✅ 写入每层的权重直方图
        for i, weights in enumerate(model.coefs_):
            writer.add_histogram(f"Weights/layer_{i}", weights, epoch)

        # ✅ 写入每层的 bias 直方图
        for i, biases in enumerate(model.intercepts_):
            writer.add_histogram(f"Biases/layer_{i}", biases, epoch)

        print(f"Epoch {epoch+1}/{max_iter} - MSE: {mse:.6f}, Loss: {model.loss_:.6f}")

    writer.close()

    # 6. 保存模型
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/train_mlp_model.joblib")

if __name__ == "__main__":
    main()
