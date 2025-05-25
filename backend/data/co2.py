import paho.mqtt.client as mqtt
import re

# 连接回调函数
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # 连接成功后，订阅主题
    client.subscribe("co2/data")

# 消息回调函数
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()}")
    # 解析消息
    message = msg.payload.decode()
    # 使用正则表达式提取数据
    match = re.split(r',', message)
    print(match)
    # 处理数据
    time = match[0]
    co2 = match[1]
    print(f"Time: {time}, CO2: {co2}")
    # 存储到csv，格式为Timestamp,CO2_Level(ppm)
    with open('./co2_data.csv', 'a') as f:
        f.write(f"{time},{co2}\n")
        f.flush()
    print(f"message saved to co2_data.csv")
        

# 创建 MQTT 客户端
client = mqtt.Client()

# 设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接到 Mosquitto
client.connect("47.122.92.94", 1883, 60)

# 启动网络循环以处理消息
client.loop_forever()