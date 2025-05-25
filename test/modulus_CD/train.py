import json
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler
from torchviz import make_dot
import imageio
import os
import pandas as pd
import ast

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

def load_csv_constraints(csv_path):
    """
    从 CSV 文件中加载约束值，并计算二氧化碳浓度的平均值。
    CSV 文件格式：
    Timestamp,CO2_Level(ppm)
    2025-04-25 12:25:50,491
    """
    data = pd.read_csv(csv_path)
    co2_levels = data["CO2_Level(ppm)"].values
    co2_mean = co2_levels.mean()
    return co2_mean

class ComplexBlock(nn.Module):
    def __init__(self, layer_size, dropout_rate=0.1):
        super(ComplexBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.LayerNorm(layer_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        return out + residual

class UltraComplexFullyConnected(nn.Module):
    def __init__(self, in_features=2, out_features=3, num_blocks=6, layer_size=512, dropout_rate=0.1):
        super(UltraComplexFullyConnected, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, layer_size),
            nn.LayerNorm(layer_size),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            ComplexBlock(layer_size, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(layer_size)
        self.output_layer = nn.Linear(layer_size, out_features)

        self._init_weights()

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.output_layer(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def trainer(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    log = PythonLogger(name="conv-diff")
    log.file_logging()
    log.info("Starting training...")
    csv_path = cfg.csv_path
    x_coords = cfg.x_coords
    y_coords = cfg.y_coords
    source_position = tuple(ast.literal_eval(cfg.source_position))
    source_strength = float(cfg.source_strength)
    print(type(source_position))

    boundary, inlet, outlet = load_geojson_geometry("room.geojson")
    (min_x, min_y), (max_x, max_y) = boundary
    print("Loaded boundary from GeoJSON:", boundary)
    print("Loaded inlet & outlet from GeoJSON:", inlet, outlet)

    # 加载 CSV 数据约束
    if csv_path:
        co2_mean = load_csv_constraints(csv_path)
        co2_mean = co2_mean / 1000.0
        print(f"Loaded CO2 mean value from CSV: {co2_mean}")
    else:
        co2_mean = None

    if x_coords is None or y_coords is None:
        raise ValueError("x_coords and y_coords must be provided when using CSV constraints.")

    height = max_y - min_y
    width = max_x - min_x
    rec = Rectangle((min_x, min_y), (max_x, max_y))

    # model = FullyConnected(in_features=2, out_features=3, num_layers=6, layer_size=512).to(dist.device)
    model = UltraComplexFullyConnected(in_features=2, out_features=3).to(dist.device)

    eqn = AdvectionDiffusion(T="c", D=0.01, rho=1.0, dim=2, time=False)
    phy_inf = PhysicsInformer(
        required_outputs=["advection_diffusion_c"],
        equations=eqn,
        grad_method="autodiff",
        device=dist.device,
    )

    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.99998**step)

    x = np.linspace(min_x, max_x, 256)
    y = np.linspace(min_y, max_y, 256)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xx, yy = torch.from_numpy(xx).float().to(dist.device), torch.from_numpy(yy).float().to(dist.device)

    bc_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=1,
        num_points=1000,
        sample_type="surface",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y"],
    )

    interior_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=1,
        num_points=4000,
        sample_type="volume",
        device=dist.device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )

    for i in range(1000):
        for bc_data, int_data in zip(bc_dataloader, interior_dataloader):

            optimizer.zero_grad()

            # 加载 CSV 数据约束
            csv_loss = 0.0
            if csv_path:
                for x, y in zip(x_coords, y_coords):
                    x_val = torch.tensor([x], device=dist.device).reshape(-1, 1)
                    y_val = torch.tensor([y], device=dist.device).reshape(-1, 1)
                    expected_co2 = torch.tensor([co2_mean], device=dist.device).float()

                    # 模型预测值
                    predicted_co2 = model(torch.cat([x_val, y_val], dim=1))

                    # 计算约束损失
                    csv_loss += torch.mean((predicted_co2 - expected_co2) ** 2)

            # 处理边界条件
            left_wall = {}
            right_wall = {}
            bottom_wall = {}
            top_wall = {}
            y_vals = bc_data[0]["y"]
            x_vals = bc_data[0]["x"]

            # 左侧墙：x == min_x
            mask_left_wall = x_vals == min_x
            # 右侧墙：x == max_x
            mask_right_wall = x_vals == max_x
            # 底层：y == min_y
            mask_bottom_wall = y_vals == min_y
            # 顶层：y == max_y
            mask_top_wall = y_vals == max_y

            # 进口边界
            mask_inlet = (
                (bc_data[0]["x"] >= inlet[0][0]) & (bc_data[0]["x"] <= inlet[1][0]) & 
                (bc_data[0]["y"] >= min(inlet[0][1], inlet[1][1])) & (bc_data[0]["y"] <= max(inlet[0][1], inlet[1][1]))
            )

            # 出口边界
            mask_outlet = (
                (bc_data[0]["x"] >= outlet[0][0]) & (bc_data[0]["x"] <= outlet[1][0]) & 
                (bc_data[0]["y"] >= min(outlet[0][1], outlet[1][1])) & (bc_data[0]["y"] <= max(outlet[0][1], outlet[1][1]))
            )

            # 左侧墙和右侧墙（无滑移边界）
            for k in bc_data[0].keys():
                left_wall[k] = (bc_data[0][k][mask_left_wall]).reshape(-1, 1)
                right_wall[k] = (bc_data[0][k][mask_right_wall]).reshape(-1, 1)

            # 底层和顶层（无滑移边界）
            for k in bc_data[0].keys():
                bottom_wall[k] = (bc_data[0][k][mask_bottom_wall]).reshape(-1, 1)
                top_wall[k] = (bc_data[0][k][mask_top_wall]).reshape(-1, 1)

            # 进口边界
            inlet_x_vals = bc_data[0]["x"][mask_inlet].reshape(-1, 1)
            inlet_y_vals = bc_data[0]["y"][mask_inlet].reshape(-1, 1)
            inlet_velocity_u = torch.ones_like(inlet_x_vals) * 1.0  
            inlet_velocity_v = torch.ones_like(inlet_y_vals) * 0.0

            outlet_x_vals = bc_data[0]["x"][mask_outlet]
            outlet_y_vals = bc_data[0]["y"][mask_outlet]

            # 处理内部数据
            interior = {}
            for k, v in int_data[0].items():
                if k in ["x", "y"]:
                    requires_grad = True
                else:
                    requires_grad = False
                interior[k] = v.reshape(-1, 1).requires_grad_(requires_grad)

            # 边界条件处理（反射、进口、出口）
            coords = torch.cat([interior["x"], interior["y"]], dim=1)
            left_wall_out = model(torch.cat([left_wall["x"], left_wall["y"]], dim=1))
            right_wall_out = model(torch.cat([right_wall["x"], right_wall["y"]], dim=1))
            bottom_wall_out = model(torch.cat([bottom_wall["x"], bottom_wall["y"]], dim=1))
            top_wall_out = model(torch.cat([top_wall["x"], top_wall["y"]], dim=1))
            interior_out = model(coords)
            inlet_out = model(torch.cat([inlet_x_vals, inlet_y_vals], dim=1))

            # 边界损失（左侧墙、右侧墙、底层、顶层）
            v_left_wall = torch.mean(left_wall_out[:, 1:2] ** 2)
            u_left_wall = torch.mean(left_wall_out[:, 0:1] ** 2)

            v_right_wall = torch.mean(right_wall_out[:, 1:2] ** 2)
            u_right_wall = torch.mean(right_wall_out[:, 0:1] ** 2)

            v_bottom_wall = torch.mean(bottom_wall_out[:, 1:2] ** 2)
            u_bottom_wall = torch.mean(bottom_wall_out[:, 0:1] ** 2)

            v_top_wall = torch.mean(top_wall_out[:, 1:2] ** 2)
            u_top_wall = torch.mean(top_wall_out[:, 0:1] ** 2)
            
            # 进口速度分量的损失
            v_inlet_loss = torch.mean((inlet_out[:, 1:2] - inlet_velocity_v) ** 2)
            u_inlet_loss = torch.mean((inlet_out[:, 0:1] - inlet_velocity_u) ** 2)

            # 处理内部约束
            phy_loss_dict = phy_inf.forward(
                {
                    "coordinates": coords,
                    "u": interior_out[:, 0:1],
                    "v": interior_out[:, 1:2],
                    "c": interior_out[:, 2:3],
                }
            )

            cont = phy_loss_dict["advection_diffusion_c"] * interior["sdf"]

            x_min, x_max, y_min, y_max = source_position

            initial_condition = torch.where(
                (interior["x"] > x_min) & (interior["x"] < x_max) &
                (interior["y"] > y_min) & (interior["y"] < y_max),
                source_strength, 0.0
            )
            
            # 总损失计算
            phy_loss = (
                1 * torch.mean(cont**2)
                + u_left_wall + v_left_wall
                + u_right_wall + v_right_wall
                + u_bottom_wall + v_bottom_wall
                + u_top_wall + v_top_wall
                + u_inlet_loss
                + v_inlet_loss
                + torch.mean((interior_out[:, 2:3] - initial_condition)**2)
                + csv_loss
            )
            print(phy_loss.item())
            phy_loss.backward()
            optimizer.step()
            scheduler.step()

            # if i == 0:
            #     pic_x = torch.randn(1, 2)
            #     pic_y = model(pic_x)
            #     dot = make_dot(pic_y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
            #     dot.render("pinn_graph", format="png", cleanup=True)

        with torch.no_grad():
                inf_out = model(torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))

                # 选择第三个通道（c）进行可视化
                inf_out_c = inf_out[:, 2].reshape(256, 256).detach().cpu().numpy()

                fig, ax = plt.subplots()
                im = ax.imshow(
                    inf_out_c,
                    origin="lower", extent=(min_x, max_x, min_y, max_y),
                    cmap='viridis',
                    vmin=0, vmax=1
                )

                cbar = plt.colorbar(im)
                cbar.ax.tick_params(labelsize=12)
                ax.set_title(f"c @ step {i}")
                plt.savefig(f"./outputs/cd_step_{i}.png", dpi=150)
                plt.close()
    # 生成 GIF
    images = []
    step_interval = 5

    for step in range(0, 1000, step_interval):
        filename = f"./outputs/cd_step_{step}.png"
        if os.path.exists(filename):
            try:
                img = imageio.imread(filename)
                images.append(img)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    if images:
        gif_path = "pinn_animation.gif"
        imageio.mimsave(gif_path, images, duration=50)  # 20fps
        print(f"✅ GIF saved at: {gif_path}")
    else:
        print("❌ No valid images found to create GIF.")
    print(f"GIF saved at: {gif_path}")
    torch.save(model.state_dict(), './pinn_model.pth')

if __name__ == "__main__":
    trainer()