import numpy as np
from catboost import CatBoostRegressor
import yaml
import os

def main():
    # 读取超参数
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    cat_params = params.get("catboost", {})

    # 读取训练数据
    data = np.load("data/processed_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]

    # 初始化模型
    model = CatBoostRegressor(
        iterations=cat_params.get("iterations", 1000),
        depth=cat_params.get("depth", 6),
        learning_rate=cat_params.get("learning_rate", 0.1),
        random_seed=42,
        verbose=False
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 确保模型保存目录存在
    os.makedirs("models", exist_ok=True)

    # 保存模型文件，格式为 CatBoost 专用的 .cb
    model.save_model("models/train_catboost_model.cb")

if __name__ == "__main__":
    main()
