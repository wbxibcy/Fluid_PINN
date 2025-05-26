from periphery import Serial
import csv
import time
from datetime import datetime
import rrdtool

def save_co2_to_rrd(co2_value, rrd_filepath="/home/linaro/Work/data/co2.rrd"):
    """
    将CO2数据保存到RRD数据库中

    参数:
        co2_value: 二氧化碳浓度值(ppm)
        rrd_filepath: 指定的RRD数据库文件路径(默认为当前目录下的co2.rrd)
    """
    try:
        # 更新RRD数据库
        # `N` 表示当前时间戳，co2_value 是要保存的 CO2 数据
        print(co2_value)
        rrdtool.update(rrd_filepath, f"N:{co2_value}")
        print(f"数据已成功保存到 RRD 数据库 {rrd_filepath}")
    except Exception as e:
        print(f"保存数据到 RRD 时出错: {e}")


def save_co2_data_to_csv(co2_value, filepath="/home/linaro/Work/data/co2_data.csv"):
    """
    将CO2数据添加时间戳并保存到指定的CSV文件
    
    参数:
        co2_value: 二氧化碳浓度值(ppm)
        filepath: 指定的CSV文件路径(默认为当前目录下的co2_data.csv)
    """
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 准备数据行
    data_row = [timestamp, co2_value]
    
    try:
        # 尝试打开文件检查是否存在
        with open(filepath, 'r') as f:
            pass
        file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    # 写入数据
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(["Timestamp", "CO2_Level(ppm)"])
        
        # 写入数据行
        writer.writerow(data_row)
    
    print(f"数据已成功保存到 {filepath}")

    
def extract_co2_value(data):
    try:
        # 检查帧头 ( 0xfe  a6)
        if data[0] != 0xfe or data[1] != 0xa6:
            raise ValueError("Invalid frame header")
        
        # 获取数据长度
        data_len = data[2]
        
        # 检查数据长度是否合理
        if len(data) != data_len + 5:  # 帧头2字节+长度1字节+command(1 bytes)+数据+CS
            raise ValueError("Data length mismatch")
        
        # 解析CO2值 (假设在位置5-6，小端格式)
        co2_value = (data[4]<<8) | data[5] # ppm
        print(f"receive data[4]:{data[4]}")
        print(f"receive data[5]:{data[5]}")
        
        return co2_value
    
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None


# Open /dev/ttyS0 with baudrate 1200, and defaults of 8N1, no flow control
sampling_interval = 300  # 5分钟 = 300秒
while True:
    # 0. open serial device
    serial = Serial("/dev/ttyS0", 1200)
    # 1. send command to read co2
    cmd_read_co2_hex = [0xfe,0xa6,0x00,0x01,0xa7]

    print(f"send hex:{bytes(cmd_read_co2_hex).hex()}")
    serial.write(bytes(cmd_read_co2_hex))

    # 2. Read up to 32 bytes with 500ms timeout
    buf = serial.read(64, 0.5)

    # printf receive data
    print(f"receive hex:{buf.hex()}")

    # 3. extract co2 ppm
    co2_value = extract_co2_value(buf)

    # 4. write data to csv file
    if co2_value is not None:
        print(f"co2: {co2_value} ppm")
        save_co2_data_to_csv(co2_value)
        save_co2_to_rrd(co2_value)
    else:
        print("Failed to extract CO2 value from data")
        
    # 5.wait 5 mins
    time.sleep(sampling_interval)

    # 6.close serial device
    serial.close()

