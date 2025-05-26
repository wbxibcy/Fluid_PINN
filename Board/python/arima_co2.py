import pandas as pd
import numpy as np
import json
import os
from pmdarima import auto_arima
import joblib
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 读取 CO2 数据
def load_co2_data(filepath):
    df = pd.read_csv(filepath)
    return df['CO2_Level(ppm)'].values

# 读取阈值配置文件
def load_threshold_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config['lower_threshold'], config['upper_threshold']

# 自动选择 ARIMA 模型并进行预测
def auto_arima_predict(values, steps=1):
    model = auto_arima(values, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    model_path = "arima_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存到 {model_path}")
    forecast = model.predict(n_periods=steps)
    return forecast[0]

# 主程序
if __name__ == "__main__":
    # 配置文件路径
    config_path = "config.json"
    # CO2 数据路径
    data_filepath = os.path.join("../data", "co2_data.csv")

    # 加载数据和配置
    co2_values = load_co2_data(data_filepath)
    lower_threshold, upper_threshold = load_threshold_config(config_path)

    # 使用自动 ARIMA 模型进行预测
    predicted_co2 = auto_arima_predict(co2_values)

    # 告警逻辑
    if predicted_co2 < lower_threshold:
        print(f"⚠️ 警告: 预测的 CO2 值 ({predicted_co2:.2f} ppm) 低于阈值 {lower_threshold} ppm!")
    elif predicted_co2 > upper_threshold:
        print(f"⚠️ 警告: 预测的 CO2 值 ({predicted_co2:.2f} ppm) 高于阈值 {upper_threshold} ppm!")
    else:
        print(f"✅ CO2 值 ({predicted_co2:.2f} ppm) 在正常范围内 ({lower_threshold} - {upper_threshold} ppm)")

