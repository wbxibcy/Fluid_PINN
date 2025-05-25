import numpy as np
import os
import json
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import shape, Point, Polygon, LineString, Point
import tempfile
import shutil
import uuid

def compute_stable_dt(dx, dy, u, v, D, safe_factor=0.5):
    epsilon = 1e-10
    dt_adv = 1.0 / (abs(u)/dx + abs(v)/dy + epsilon)
    dt_diff = 0.5 / (D * (1.0/dx**2 + 1.0/dy**2) + epsilon)
    return safe_factor * min(dt_adv, dt_diff)

def load_geojson(geojson_data):
    """
    接收 GeoJSON 数据并返回解析后的数据。
    """
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)
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
    x_base = np.linspace(minx, maxx, nx)
    y_base = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x_base, y_base)
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
    line = LineString(line_coords)
    p = Point(point)
    return line.distance(p) < tol

def generate_fvm_animation(geojson_data, D, steps, u, v, source_position, source_strength):
    """
    使用有限体积法（FVM）模拟扩散过程并生成动画 GIF。
    """
    # 加载 GeoJSON 数据
    geojson_data = load_geojson(geojson_data)

    # 提取边界和障碍物
    boundary, holes = extract_polygon_data(geojson_data)

    # 获取边界条件
    house_boundary, inlet_coords, outlet_coords = get_boundary_conditions(geojson_data)

    # 创建结构化网格
    X, Y, mask = create_structured_grid(boundary, holes, nx=300, ny=300)
    Ny, Nx = X.shape

    # 初始化网格和参数
    points = np.column_stack([X.flatten(), Y.flatten()])
    cells = np.array([
        [j * Nx + i, j * Nx + i + 1, (j + 1) * Nx + i + 1, (j + 1) * Nx + i]
        for j in range(Ny - 1) for i in range(Nx - 1)
    ])
    x_coords = np.unique(points[:, 0])
    y_coords = np.unique(points[:, 1])
    Nx, Ny = len(x_coords) - 1, len(y_coords) - 1
    dx, dy = x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]
    dt = compute_stable_dt(dx, dy, u, v, D)

    # 定义初始源区域
    x_min, x_max, y_min, y_max = source_position
    # 将源区域的物理坐标转换为网格坐标
    source_x_min = int(np.floor((x_min - x_coords[0]) / dx))
    source_x_max = int(np.floor((x_max - x_coords[0]) / dx))
    source_y_min = int(np.floor((y_min - y_coords[0]) / dy))
    source_y_max = int(np.floor((y_max - y_coords[0]) / dy))

    # 定义初始源区域
    source_region = (
        slice(source_y_min, source_y_max),
        slice(source_x_min, source_x_max)
    )

    pad = 1
    C = np.zeros((Ny, Nx))
    C[source_region] = source_strength
    C = np.pad(C, ((0, 0), (pad, pad)), mode='edge')

    # 使用临时文件夹存储 GIF 帧
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")

        # 模拟扩散过程
        for step in range(steps):
            print(step)
            C_new = C.copy()
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    adv = fvm_advection(C, i, j, dx, dy, u, v)
                    diff_x = D * (C[j, i + 1] - 2 * C[j, i] + C[j, i - 1]) / dx**2
                    diff_y = D * (C[j + 1, i] - 2 * C[j, i] + C[j - 1, i]) / dy**2
                    C_new[j, i] = C[j, i] + dt * (adv + diff_x + diff_y)

            # 处理出口条件
            for j in range(Ny):
                for i in range(Nx):
                    if point_on_line_segment((x_coords[i], y_coords[j]), outlet_coords):
                        distance = np.abs(x_coords[i] - np.mean([coord[0] for coord in outlet_coords]))
                        C_new[j, i] = max(0.0, C_new[j, i] - distance * 0.1)

            # 更新边界条件
            C_new[0, :] = C_new[1, :]
            C_new[-1, :] = C_new[-2, :]
            C_new[:, :pad] = C_new[:, pad:pad+1]
            C_new[:, pad+Nx:] = C_new[:, pad+Nx-1:pad+Nx]
            C = C_new
            C_new[source_region] = 1.0

            # 绘制并保存每一帧
            C_core = C[:, pad-1:pad+Nx-1]
            cmap = sns.color_palette("coolwarm", as_cmap=True)
            fig, ax = plt.subplots(figsize=(10, 10))
            x = np.linspace(0, Nx * dx, Nx)
            y = np.linspace(0, Ny * dy, Ny)
            X, Y = np.meshgrid(x, y)
            c = ax.contourf(X, Y, C_core, levels=200, cmap=cmap, vmin=0, vmax=1)
            cbar = plt.colorbar(c, ax=ax, ticks=np.linspace(0, 1, 6))
            cbar.set_label("CO2")
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
            plt.savefig(f"{temp_dir}/frame_{step:04d}.png", dpi=200)
            plt.close()

        # 分批生成 GIF
        unique_id = uuid.uuid4().hex
        gif_path = os.path.join("./outputs", f"{unique_id}.gif")
        with imageio.get_writer(gif_path, mode='I', fps=10) as writer:
            for step in range(steps):
                frame_path = os.path.join(temp_dir, f"frame_{step:04d}.png")
                if os.path.exists(frame_path):
                    writer.append_data(imageio.imread(frame_path))
                    os.remove(frame_path)
        print(f"GIF saved at: {gif_path}")

    return gif_path

if __name__ == "__main__":
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": { "type": "boundary" },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]
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

    gif_path = generate_fvm_animation(
        geojson_data=geojson_data,
        D=0.01,
        steps=200,
        u=0.5,
        v=0.5,
        source_position = (2.0, 4.0, 3.0, 5.0),
        source_strength = 1.0
    )
    print(f"生成的GIF路径: {gif_path}")