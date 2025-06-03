import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # 读取原始数据
    df = pd.read_csv("data/insurance.csv")

    # One-Hot 编码分类变量
    df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

    # 分离特征和标签
    X = df_encoded.drop("charges", axis=1).values
    y = df_encoded["charges"].values

    # 按 60/40 划分训练和验证
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # 标准化（仅使用训练集拟合）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_full_scaled = scaler.transform(X)

    # 保存划分后的数据
    np.savez("data/processed_split.npz",
             X_train=X_train_scaled, y_train=y_train,
             X_val=X_val_scaled, y_val=y_val)

    # 保存全集数据
    np.savez("data/processed_full.npz", X=X_full_scaled, y=y)

if __name__ == "__main__":
    main()
