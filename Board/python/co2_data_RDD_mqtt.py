# -*- coding: utf-8 -*-
import rrdtool
import paho.mqtt.client as mqtt
import time
from datetime import datetime

def fetch_rrd_data(rrd_filepath="/home/linaro/Work/data/co2.rrd", start_time=None, end_time=None):
    try:
        # 获取当前时间，如果没有提供开始时间和结束时间，则默认获取过去30分钟的数据
        info = rrdtool.info(rrd_filepath)
        print(info)
        if start_time is None:
            start_time = int(time.time()) - 1800  # 默认过去30分钟数据
        if end_time is None:
            end_time = int(time.time())

        # 读取 RRD 数据
        print(f"Fetching data from {start_time} to {end_time}")
        result = rrdtool.fetch(rrd_filepath, "AVERAGE", "--start", str(start_time), "--end", str(end_time))
        timestamps = result[0]  # 时间戳列表
        data = result[2]  # 数据列表

        return timestamps, data
    except Exception as e:
        print(f"Error fetching RRD data: {e}")
        return [], []

def calculate_average(data):
    """
    计算数据的平均值，忽略无效数据（例如值为 `NaN` 或 `None`）
    """
    valid_data = [value[0] for value in data if value[0] is not None]  # 提取每个数据元组中的实际值
    print(valid_data)

    if not valid_data:  # 如果没有有效数据，返回 0
        return 0

    return sum(valid_data) / len(valid_data)  # 计算平均值

def send_to_mqtt(co2_avg, mqtt_broker="47.122.92.94", mqtt_topic="co2/data"):
    if co2_avg is None:
        print("No valid data for MQTT publishing.")
        return

    # 创建MQTT客户端
    client = mqtt.Client()
    client.connect(mqtt_broker, 1883, 60)

    # 获取当前时间戳并创建消息
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 发送格式：时间戳 + 平均值
    message = f"{timestamp},{co2_avg}"

    # 发布消息到MQTT
    client.publish(mqtt_topic, message)
    print(f"Published: {message}")

    client.disconnect()

if __name__ == "__main__":
    # 每30分钟读取一次并发送数据
    while True:
        # 获取过去30分钟的数据
        timestamps, data = fetch_rrd_data()
        print(data)

        if len(timestamps) > 0 and len(data) > 0:
            # 计算过去30分钟的数据的平均值
            co2_avg = calculate_average(data)
            # 发送平均值到MQTT
            send_to_mqtt(co2_avg)

        # 等待30分钟
        time.sleep(1800)

