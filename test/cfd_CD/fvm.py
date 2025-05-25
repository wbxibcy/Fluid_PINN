import numpy as np
import os
import matplotlib.pyplot as plt
import json
import imageio
import seaborn as sns

# 参数设置
D = 0.01
u, v = 2.0, 1.0
steps = 300
pad = 1


def compute_stable_dt(dx, dy, u, v, D, safe_factor=0.5):
    # 避免除以0的小偏移
    epsilon = 1e-10

    # 对流主导时的CFL稳定条件
    dt_adv = 1.0 / (abs(u)/dx + abs(v)/dy + epsilon)
    # 扩散主导时的稳定条件（显式中心差分）
    dt_diff = 0.5 / (D * (1.0/dx**2 + 1.0/dy**2) + epsilon)

    # 使用较小的时间步长并乘以安全因子
    return safe_factor * min(dt_adv, dt_diff)

# 加载结构化网格
def load_structured_grid(npz_file):
    data = np.load(npz_file)
    X, Y, mask = data['X'], data['Y'], data['mask']
    Ny, Nx = X.shape
    points = np.column_stack([X.flatten(), Y.flatten()])
    cells = [
        [j * Nx + i, j * Nx + i + 1, (j + 1) * Nx + i + 1, (j + 1) * Nx + i]
        for j in range(Ny - 1) for i in range(Nx - 1)
    ]
    return points, np.array(cells), mask

# 读取 GeoJSON 边界条件
def load_geojson(geojson_file):
    with open(geojson_file, 'r') as f:
        return json.load(f)

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

def point_on_line_segment(point, line_coords, tol=1e-2):
    from shapely.geometry import LineString, Point
    line = LineString(line_coords)
    p = Point(point)
    return line.distance(p) < tol

# 对流项
def fvm_advection(C, i, j, dx, dy, u, v):
    adv_x = -u * (C[j, i] - C[j, i - 1]) / dx if u > 0 else -u * (C[j, i + 1] - C[j, i]) / dx
    adv_y = -v * (C[j, i] - C[j - 1, i]) / dy if v > 0 else -v * (C[j + 1, i] - C[j, i]) / dy
    return adv_x + adv_y

# 初始化
npz_file = './meshes/structured_grid.npz'
geojson_file = './modeling/room.geojson'
points, cells, mask = load_structured_grid(npz_file)
boundary_conditions = load_geojson(geojson_file)
house_boundary, inlet_coords, outlet_coords = get_boundary_conditions(boundary_conditions)

# 网格信息
x_coords = np.unique(points[:, 0])
y_coords = np.unique(points[:, 1])
Nx, Ny = len(x_coords) - 1, len(y_coords) - 1
dx, dy = x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]

dt = compute_stable_dt(dx, dy, u, v, D)

# 初始浓度场
C = np.zeros((Ny, Nx))
# 中心点坐标
center_x = Nx // 2
center_y = Ny // 2

# 源大小（例如 10x10）
source_half_width = 5
source_half_height = 5

source_region = (
    slice(center_y - source_half_height, center_y + source_half_height),
    slice(center_x - source_half_width, center_x + source_half_width)
)

# 设置浓度为 1 的中心区域
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
print(f"✅ GIF 已生成：{gif_path}")
