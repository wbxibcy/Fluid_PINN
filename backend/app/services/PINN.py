import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from omegaconf import DictConfig
from torch.optim import Adam, lr_scheduler
import pandas as pd
from omegaconf import OmegaConf
import os
import uuid

def load_geojson_geometry(geojson_data):
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)

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


def trainer(
    geojson_data: dict,
    cfg: DictConfig = None,
    csv_path: str = None,
    x_coords: list = None,
    y_coords: list = None,
    source_position: tuple = None,
    source_strength: float = 1.0,
    patience: int = 100
) -> None:
    if cfg is None:
        cfg = OmegaConf.load("./app/services/config.yaml")
    # if x_coords is None or y_coords is None:
    #     raise ValueError("x_coords and y_coords must be provided when using CSV constraints.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = PythonLogger(name="conv-diff")
    log.file_logging()

    boundary, inlet, outlet = load_geojson_geometry(geojson_data)
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

    height = max_y - min_y
    width = max_x - min_x
    rec = Rectangle((min_x, min_y), (max_x, max_y))

    model = FullyConnected(in_features=2, out_features=3, num_layers=6, layer_size=512).to(device)

    eqn = AdvectionDiffusion(T="c", D=0.01, rho=1.0, dim=2, time=False)
    phy_inf = PhysicsInformer(
        required_outputs=["advection_diffusion_c"],
        equations=eqn,
        grad_method="autodiff",
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.99998**step)

    best_loss = float("inf")  # 最佳损失，初始化为无穷大
    patience_counter = 0  # 连续多少步没有改善
    stop_training = False  # 控制是否停止训练

    x = np.linspace(min_x, max_x, 256)
    y = np.linspace(min_y, max_y, 256)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xx, yy = torch.from_numpy(xx).float().to(device), torch.from_numpy(yy).float().to(device)

    bc_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=1,
        num_points=1000,
        sample_type="surface",
        device=device,
        num_workers=1,
        requested_vars=["x", "y"],
    )

    interior_dataloader = GeometryDatapipe(
        geom_objects=[rec],
        batch_size=1,
        num_points=4000,
        sample_type="volume",
        device=device,
        num_workers=1,
        requested_vars=["x", "y", "sdf"],
    )

    for i in range(1500):
        for bc_data, int_data in zip(bc_dataloader, interior_dataloader):

            optimizer.zero_grad()

            # 加载 CSV 数据约束
            csv_loss = torch.tensor(0.0)
            if csv_path:
                if x_coords is not None and y_coords is not None:
                    csv_loss = 0.0
                    for x, y in zip(x_coords, y_coords):
                        x_val = torch.tensor([x], device=device).float().reshape(-1, 1)
                        y_val = torch.tensor([y], device=device).float().reshape(-1, 1)
                        expected_co2 = torch.tensor([co2_mean], device=device).float()

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

            # 出口边界（自然流出）：保持无强制速度条件
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

            if source_position is None:
                source_position = (
                    min_x + width / 4,  # x_min
                    max_x - width / 4,  # x_max
                    min_y + height / 4, # y_min
                    max_y - height / 4  # y_max
                )

            x_min, x_max, y_min, y_max = source_position

            initial_condition = torch.where(
                (interior["x"] > x_min) & (interior["x"] < x_max) &
                (interior["y"] > y_min) & (interior["y"] < y_max),
                source_strength, 0.0
            )
            
            print("cont loss:", torch.mean(cont**2).item())
            print("boundary losses:", u_left_wall.item(), v_left_wall.item(), u_right_wall.item(), v_right_wall.item())
            print("inlet losses:", u_inlet_loss.item(), v_inlet_loss.item())
            print("initial condition loss:", torch.mean((interior_out[:, 2:3] - initial_condition)**2).item())
            print("csv_loss:", csv_loss.item())

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

            # 早停策略
            if phy_loss < best_loss:
                best_loss = phy_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at step {i}.")
                stop_training = True
                break

            phy_loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradient!")

            optimizer.step()
            scheduler.step()

            print(f"Step {i}, Loss: {phy_loss.item()}")

    # 保存可视化图片
    with torch.no_grad():
        inf_out = model(torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))
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
        unique_id = uuid.uuid4().hex
        filename = f"./outputs/{unique_id}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved final image to {filename}")
    
    return filename

if __name__ == "__main__":
    geodata = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": { "type": "boundary" },
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0]]]
        }
      },
      {
        "type": "Feature",
        "properties": { "type": "inlet" },
        "geometry": {
          "type": "LineString",
          "coordinates": [[0, 4.5], [0, 5.5]]
        }
      },
      {
        "type": "Feature",
        "properties": { "type": "outlet" },
        "geometry": {
          "type": "LineString",
          "coordinates": [[10, 4.5], [10, 5.5]]
        }
      }
    ]
  }
    csv_path="./data/co2_data.csv"
    x_coords=[1.0]
    y_coords=[4.0]
    source_position = (2.0, 4.0, 3.0, 5.0)
    source_strength = 1.0
    trainer(geojson_data=geodata, csv_path=csv_path, x_coords=x_coords, y_coords=y_coords, source_position=source_position, source_strength=source_strength)
