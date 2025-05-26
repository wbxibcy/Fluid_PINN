#!/bin/bash

echo "🔧 开始执行 CO2 预测训练流程"

# 1. 运行 Q-Learning 阶段
echo "🚀 正在运行 q_learning.py..."
python3 q_learning.py
if [ $? -ne 0 ]; then
    echo "❌ q_learning.py 执行失败，终止流程。"
    exit 1
fi

# 2. 运行 ARIMA 模型训练阶段
echo "📈 正在运行 arima_co2.py..."
python3 arima_co2.py
if [ $? -ne 0 ]; then
    echo "❌ arima_co2.py 执行失败，终止流程。"
    exit 1
fi

echo "✅ 模型训练和保存流程执行完毕。"

