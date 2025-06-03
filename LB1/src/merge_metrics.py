import json
import os
import pandas as pd
from tabulate import tabulate  # 新增库，需要提前安装：pip install tabulate

def parse_filename(filename):
    parts = filename.replace(".json", "").split("_")
    if parts[0] == "evaluate":
        model = parts[1]
        dataset = "val"
    elif parts[0] == "full":
        model = parts[1]
        dataset = "full"
    else:
        model = parts[0]
        dataset = "unknown"
    return model, dataset

def main():
    metrics_dir = "metrics"
    rows = []

    for file in os.listdir(metrics_dir):
        if file.endswith(".json"):
            filepath = os.path.join(metrics_dir, file)
            with open(filepath, "r") as f:
                data = json.load(f)

            model, dataset = parse_filename(file)

            row = {
                "model": model,
                "dataset": dataset,
                "mse": data.get("MSE", None),
                "mae": data.get("MAE", None),
                "r2": data.get("R2", None)
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    output_csv = "results/all_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Merged metrics written to {output_csv}")

    # 使用 tabulate 打印漂亮的表格
    print("\n合并指标表格预览:\n")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

if __name__ == "__main__":
    main()
