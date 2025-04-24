"""
@File  :data_parse.py
@Author:Ezra Zephyr
@Date  :2025/4/1721:13
@Desc  :
"""
import pandas as pd

# 读取旧的 CSV 文件
input_file_path = '../stock_data/edf_stocks.csv'  # 请替换成你旧文件的路径
df = pd.read_csv(input_file_path)

# 查看原始数据
print("Original data:")
print(df.head())

# 创建新的列 'PriceRange'，表示 'HIGH' 和 'LOW' 的差值
df["PriceRange"] = df["HIGH"] - df["LOW"]

# 创建一个新的 'datetime' 列，合并 'DATE' 和 'TIME' 列
df["datetime"] = df["DATE"] + " " + df["TIME"]
df["datetime"] = pd.to_datetime(df["datetime"])

# 将 'datetime' 列转换为时间戳（单位为秒）
df["timestamp"] = df["datetime"].astype(int) / 10**9  # 将时间戳转换为秒

# 排序数据，根据 'datetime' 列进行排序
df = df.sort_values(by=["datetime"])

# 删除原始的 'DATE' 和 'TIME' 列，因为它们已经合并成了 'datetime' 列
df = df.drop(columns=["DATE", "TIME"])

# 选择需要保存的列
df = df[["timestamp", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "PriceRange"]]

# 保存为新的 CSV 文件
output_file_path = '../stock_data/processed_stocks.csv'  # 请替换成你希望保存的新文件路径
df.to_csv(output_file_path, index=False)

print(f"Processed data saved as '{output_file_path}'")
