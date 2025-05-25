import paho.mqtt.client as mqtt

# 连接回调函数
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # 连接成功后，订阅主题
    client.subscribe("co2/data")

# 消息回调函数
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()}")

# 创建 MQTT 客户端
client = mqtt.Client()

# 设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接到 Mosquitto
client.connect("47.122.92.94", 1883, 60)

# 启动网络循环以处理消息
client.loop_forever()