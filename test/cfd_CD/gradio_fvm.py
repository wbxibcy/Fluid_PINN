import numpy as np
import os
import json
import gradio as gr
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import shape, Point, Polygon

# 参数设置 (可以在接口中调整)
def compute_stable_dt(dx, dy, u, v, D, safe_factor=0.5):
    # 避免除以0的小偏移
    epsilon = 1e-10

    # 对流主导时的CFL稳定条件
    dt_adv = 1.0 / (abs(u)/dx + abs(v)/dy + epsilon)
    # 扩散主导时的稳定条件（显式中心差分）
    dt_diff = 0.5 / (D * (1.0/dx**2 + 1.0/dy**2) + epsilon)

    # 使用较小的时间步长并乘以安全因子
    return safe_factor * min(dt_adv, dt_diff)

def load_geojson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    return geojson_data

def extract_polygon_data(geojson_data):
    boundary = None
    holes = []

    for feature in geojson_data["features"]:
        geom = shape(feature["geometry"])
        ftype = feature["properties"].get("type")
        if ftype == "boundary":
            boundary = geom
        elif ftype == "obstacle":
            holes.append(geom)

    if boundary is None:
        raise ValueError("GeoJSON must contain a 'boundary' polygon")

    return boundary, holes

# 边界条件提取
def get_boundary_conditions(boundary_conditions):
    house_boundary = None
    inlet_coords = []
    outlet_coords = []
    for feature in boundary_conditions["features"]:
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]
        if feature["properties"]["type"] == "boundary":
            house_boundary = coords[0]
        elif feature["properties"]["type"] == "inlet" and geom_type == "LineString":
            inlet_coords.extend(coords)
        elif feature["properties"]["type"] == "outlet" and geom_type == "LineString":
            outlet_coords.extend(coords)
    return house_boundary, inlet_coords, outlet_coords

def create_structured_grid(boundary: Polygon, holes, nx, ny):
    minx, miny, maxx, maxy = boundary.bounds

    # 均匀结构网格
    x_base = np.linspace(minx, maxx, nx)
    y_base = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x_base, y_base)

    # 掩膜只保留在区域内的点
    mask = np.zeros(X.shape, dtype=bool)
    for j in range(Y.shape[0]):
        for i in range(X.shape[1]):
            pt = Point(X[j, i], Y[j, i])
            if boundary.contains(pt) and all(not hole.contains(pt) for hole in holes):
                mask[j, i] = True

    return X, Y, mask

def fvm_advection(C, i, j, dx, dy, u, v):
    adv_x = -u * (C[j, i] - C[j, i - 1]) / dx if u > 0 else -u * (C[j, i + 1] - C[j, i]) / dx
    adv_y = -v * (C[j, i] - C[j - 1, i]) / dy if v > 0 else -v * (C[j + 1, i] - C[j, i]) / dy
    return adv_x + adv_y

def point_on_line_segment(point, line_coords, tol=1e-2):
    from shapely.geometry import LineString, Point
    line = LineString(line_coords)
    p = Point(point)
    return line.distance(p) < tol

def generate_fvm_animation(geojson_file, D, steps, u, v, center_x, center_y, source_half_width, source_half_height):
    geojson_data = load_geojson(geojson_file)
    boundary, holes = extract_polygon_data(geojson_data)
    house_boundary, inlet_coords, outlet_coords = get_boundary_conditions(geojson_data)

    # 创建结构网格
    X, Y, mask = create_structured_grid(boundary, holes, nx=300, ny=300)
    
    Ny, Nx = X.shape
    points = np.column_stack([X.flatten(), Y.flatten()])
    cells = np.array([
        [j * Nx + i, j * Nx + i + 1, (j + 1) * Nx + i + 1, (j + 1) * Nx + i]
        for j in range(Ny - 1) for i in range(Nx - 1)
    ])

    # 网格信息
    x_coords = np.unique(points[:, 0])
    y_coords = np.unique(points[:, 1])
    Nx, Ny = len(x_coords) - 1, len(y_coords) - 1
    dx, dy = x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]

    dt = compute_stable_dt(dx, dy, u, v, D)

    # 初始浓度场
    source_region = (
        slice(center_y - source_half_height, center_y + source_half_height),
        slice(center_x - source_half_width, center_x + source_half_width)
    )
    pad = 1
    C = np.zeros((Ny, Nx))
    # 使用用户输入的初始源位置
    C[source_region] = 1.0
    C = np.pad(C, ((0, 0), (pad, pad)), mode='edge')

    # 输出目录
    gif_folder = "gif"
    os.makedirs(gif_folder, exist_ok=True)

    # 主循环
    for step in range(steps):
        print(step)
        C_new = C.copy()
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                adv = fvm_advection(C, i, j, dx, dy, u, v)
                diff_x = D * (C[j, i + 1] - 2 * C[j, i] + C[j, i - 1]) / dx**2
                diff_y = D * (C[j + 1, i] - 2 * C[j, i] + C[j - 1, i]) / dy**2
                C_new[j, i] = C[j, i] + dt * (adv + diff_x + diff_y)

        # 出口边界条件
        for j in range(Ny):
            for i in range(Nx):
                if point_on_line_segment((x_coords[i], y_coords[j]), outlet_coords):
                    distance = np.abs(x_coords[i] - np.mean([coord[0] for coord in outlet_coords]))
                    C_new[j, i] = max(0.0, C_new[j, i] - distance * 0.1)

        # 处理反射边界（顶部、底部、左边界、右边界）
        C_new[0, :] = C_new[1, :]  # 顶部反射
        C_new[-1, :] = C_new[-2, :]  # 底部反射
        C_new[:, :pad] = C_new[:, pad:pad+1]  # 左边界反射
        C_new[:, pad+Nx:] = C_new[:, pad+Nx-1:pad+Nx]  # 右边界反射

        C = C_new
        C_new[source_region] = 1.0
        C_core = C[:, pad-1:pad+Nx-1]

        # 绘图
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        x = np.linspace(0, Nx * dx, Nx)
        y = np.linspace(0, Ny * dy, Ny)
        X, Y = np.meshgrid(x, y)
        c = ax.contourf(X, Y, C_core, levels=200, cmap=cmap, vmin=0, vmax=1)
        cbar = plt.colorbar(c, ax=ax, ticks=np.linspace(0, 1, 6))
        cbar.set_label("CO2")

        # 房屋、入口、出口边界绘制
        if house_boundary:
            house_boundary_np = np.array(house_boundary)
            ax.plot(house_boundary_np[:, 0], house_boundary_np[:, 1], color='white', linewidth=1.5, label="House Boundary")
        if inlet_coords:
            inlet_np = np.array(inlet_coords)
            ax.plot(inlet_np[:, 0], inlet_np[:, 1], 'g-', linewidth=2, label="Inlet")
        if outlet_coords:
            outlet_np = np.array(outlet_coords)
            ax.plot(outlet_np[:, 0], outlet_np[:, 1], 'r-', linewidth=2, label="Outlet")

        ax.set_title(f"Step {step}", fontsize=12)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        cbar.set_label("CO2")
        plt.tight_layout()
        plt.savefig(f"{gif_folder}/frame_{step:04d}.png", dpi=150)
        plt.close()

    # 生成 GIF 动画
    images = []
    for step in range(steps):
        filename = f"{gif_folder}/frame_{step:04d}.png"
        if os.path.exists(filename):
            images.append(imageio.imread(filename))

    gif_path = os.path.join(gif_folder, "animation.gif")
    imageio.mimsave(gif_path, images, fps=10)

    return gif_path

# Gradio 接口
def gradio_interface(geojson_file, D, steps, u, v, center_x, center_y, source_half_width, source_half_height):
    # 使用上传的 geojson 和参数生成动画
    gif_path = generate_fvm_animation(geojson_file.name, D, steps, u, v, center_x, center_y, source_half_width, source_half_height)
    return gif_path

# 创建 Gradio 接口
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload GeoJSON File"),
        gr.Number(label="Diffusion Coefficient (D)", value=0.01),
        gr.Number(label="Steps", value=20),
        gr.Number(label="Horizontal Velocity (u)", value=1.0),
        gr.Number(label="Vertical Velocity (v)", value=0.0),
        gr.Number(label="Initial Source X Position", value=50),
        gr.Number(label="Initial Source Y Position", value=150),
        gr.Number(label="Initial Source Width (source_half_width)", value=5),
        gr.Number(label="Initial Source Height (source_half_height)", value=5)
    ],
    outputs=gr.Image(type="filepath", label="Generated GIF"),
    live=True
)

iface.launch(server_port=7861)
