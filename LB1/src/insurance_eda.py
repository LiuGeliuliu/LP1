# insurance_eda.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置图像样式
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. 读取数据
df = pd.read_csv("D:/YANER/MII/LB1/data/raw/insurance.csv")

# 2. 数据基本信息
print("基本信息：")
print(df.info())
print("\n描述性统计：")
print(df.describe())
print("\n类别变量分布：")
for col in ['sex', 'smoker', 'region']:
    print(f"\n{col}:\n", df[col].value_counts())

# 3. 缺失值检查
print("\n缺失值检查：")
print(df.isnull().sum())

# 4. 数值特征的分布
num_features = ['age', 'bmi', 'children', 'charges']
for col in num_features:
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} 分布图')
    plt.show()

# 5. 类别特征分布
cat_features = ['sex', 'smoker', 'region']
for col in cat_features:
    sns.countplot(data=df, x=col)
    plt.title(f'{col} 分布图')
    plt.show()

# 6. 类别变量对目标变量的影响（箱线图）
for col in cat_features:
    sns.boxplot(x=col, y='charges', data=df)
    plt.title(f'{col} 与 charges 的关系')
    plt.show()

# 7. 数值变量与目标变量的关系（散点图）
for col in ['age', 'bmi']:
    sns.scatterplot(x=col, y='charges', hue='smoker', data=df)
    plt.title(f'{col} 与 charges 的关系（按 smoker 分组）')
    plt.show()

# 8. 相关性矩阵
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("相关系数热力图")
plt.show()

# 9. 类别特征编码准备（用于后续建模）
df_encoded = pd.get_dummies(df, drop_first=True)
print("\n编码后的特征：", df_encoded.columns.tolist())

# 10. 特征与目标变量的线性相关性（编码后）
correlation = df_encoded.corr()['charges'].sort_values(ascending=False)
print("\n特征与 charges 的相关性：\n", correlation)

# 11. 小结和特征选择建议
print("\n--- 初步结论 ---")
print("1. 'smoker' 是最强影响因素，吸烟者医疗费用远高于非吸烟者。")
print("2. 'age' 和 'bmi' 也与费用呈正相关，尤其在高 BMI 的吸烟者中。")
print("3. 'region', 'sex' 对医疗费用影响相对较小。")
print("建议选择的特征： age, bmi, children, smoker, sex, region（编码后）。")
