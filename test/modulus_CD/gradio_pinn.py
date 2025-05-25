import gradio as gr
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.models.mlp.fully_connected import FullyConnected
from PIL import Image
import io
import imageio
import os

# 加载 PyTorch 模型
def load_model(model_path):
    model = FullyConnected(in_features=2, out_features=1, num_layers=6, layer_size=512)
    model.load_state_dict(torch.load(model_path))  # 加载模型
    model.eval()  # 设置为评估模式
    return model

def load_geojson_geometry(filepath):
    with open(filepath, 'r') as f:
        geojson_data = json.load(f)

    boundary = None
    inlet = []
    outlet = []

    for feature in geojson_data['features']:
        ftype = feature['properties']['type']
        coords = feature['geometry']['coordinates']

        if ftype == 'boundary':
            flat_coords = coords[0]
            xs = [pt[0] for pt in flat_coords]
            ys = [pt[1] for pt in flat_coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            boundary = ((min_x, min_y), (max_x, max_y))
        elif ftype == 'inlet':
            inlet = coords
        elif ftype == 'outlet':
            outlet = coords

    return boundary, inlet, outlet

# 使用模型进行推理生成图像，并返回PIL图像
def generate_image_from_model(model, xx, yy, min_x, max_x, min_y, max_y, time_step):
    inf_out = model(torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))

    fig, ax = plt.subplots()
    im = ax.imshow(
        inf_out.reshape(256, 256).detach().cpu().numpy(),
        origin="lower", extent=(min_x, max_x, min_y, max_y),
        cmap='viridis',
        vmin=0, vmax=1
    )

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(f"c @ step {time_step}")

    # 保存为内存中的图像（PIL格式）
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img

# 将多个图像保存为 GIF
def generate_gif(images, gif_duration=0.1):
    gif_path = './output_animation.gif'
    imageio.mimsave(gif_path, images, duration=gif_duration)
    return gif_path

# Gradio 接口
def gradio_interface(geojson_file, time_step, gif_duration=0.1):
    # 读取模型
    model = load_model('./pinn_model.pth')
    boundary, inlet, outlet = load_geojson_geometry(geojson_file.name)
    (min_x, min_y), (max_x, max_y) = boundary

    x = np.linspace(min_x, max_x, 256)  # 输入范围 x
    y = np.linspace(min_y, max_y, 256)  # 输入范围 y
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xx, yy = torch.from_numpy(xx).float(), torch.from_numpy(yy).float()
    
    # 创建多个图像（假设每一帧是模型推理的结果）
    images = []
    for t in range(time_step):
        print(t)
        img = generate_image_from_model(model, xx, yy, min_x, max_x, min_y, max_y, t)
        
        # 将PIL图像添加到列表中
        images.append(img)
    
    # 生成 GIF
    gif_path = generate_gif(images, gif_duration)
    
    return gif_path

# Gradio 界面设置
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload GeoJSON File"),
        gr.Number(value=100, label="Steps"),
        gr.Number(value=0.1, label="GIF Duration")
    ],
    outputs=gr.Image(type="filepath", label="Generated GIF"),
    title="PyTorch Model",
    description="Upload a GeoJSON file containing geographical points, and set parameters to generate a GIF.",
    live=True
)

# 启动 Gradio 应用
if __name__ == "__main__":
    iface.launch(server_port=7862)
