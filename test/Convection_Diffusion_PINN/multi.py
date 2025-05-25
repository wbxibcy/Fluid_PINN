import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
import torch
import os


if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    print("Using GPU")
else:
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    print("Using CPU")

# 定义多个矩形区域
geom_rect1 = dde.geometry.Rectangle([0, 0], [1.5, 1.5])
geom_rect2 = dde.geometry.Rectangle([1, 1], [2, 2])
geom_rect3 = dde.geometry.Rectangle([-1.5, -1.5], [0.5, 0.5])

# 使用 CSGUnion 将它们合并成一个几何区域
geom = dde.geometry.CSGUnion(dde.geometry.CSGUnion(geom_rect1, geom_rect2), geom_rect3)

# 定义时间区域
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 扩散系数
nu = 1e-3  # 粘性系数
v_x = 0.5  # 对流速度的 x 方向分量
v_y = 0  # 对流速度的 y 方向分量

# 定义 PDE（扩散对流方程）
def pde(x, u):
    u_val = u[:, 0:1]

    u_t = dde.grad.jacobian(u, x, i=0, j=2)  # ∂u/∂t

    # 一阶导数（对流项）
    u_x = dde.grad.jacobian(u_val, x, i=0, j=0)  # ∂u/∂x
    u_y = dde.grad.jacobian(u_val, x, i=0, j=1)  # ∂u/∂y

    # 二阶导数（扩散项）
    u_xx = dde.grad.hessian(u_val, x, component=0, i=0, j=0)  # ∂²u/∂x²
    u_yy = dde.grad.hessian(u_val, x, component=0, i=1, j=1)  # ∂²u/∂y²

    convection = v_x * u_x + v_y * u_y  # 对流项
    diffusion = nu * (u_xx + u_yy)  # 扩散项

    # PDE 方程
    pde_equation = u_t + convection - diffusion

    return pde_equation

# 上边界，y=0
def boundary_t(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

# 下边界，y=0
def boundary_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

# 左边界，x=0
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

# 右边界，x=0
def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

bc_t = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary_t)
bc_b = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary_b)
bc_l = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l)
bc_r = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_r)

# 初始条件：一个扩散源
def initial_condition(x):
    return 0.1 * np.ones_like(x)[:, 0:1]

ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial,)

# 设置数据
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_l, bc_r, bc_b, bc_t, ic],
    num_domain=1000,
    num_boundary=500,
    num_initial=500,
    num_test=1000
)

layer_size = [3] + [50] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4)

# 设置早停回调
early_stopping = dde.callbacks.EarlyStopping(monitor="loss_test", patience=1000)

losshistory, train_state = model.train(iterations=5000, display_every=1000, callbacks=[early_stopping])
dde.saveplot(losshistory, train_state, isplot=False, issave=True)

print("Drawing...")
# 计算x和y离散化的范围，适应新的复合几何区域
x1 = np.linspace(-2, 2, num=1000, endpoint=True).flatten()  # x的范围根据复合几何区域的最大范围设定
y1 = np.linspace(-2, 2, num=1000, endpoint=True).flatten()  # y的范围根据复合几何区域的最大范围设定
xx1, yy1 = np.meshgrid(x1, y1)
x = xx1.flatten()
y = yy1.flatten()

# 只保留在复合几何区域内的点
mask = np.array([geom.inside(np.array([xi, yi])) for xi, yi in zip(x, y)])
x = x[mask]
y = y[mask]

# 时间上取20个时间步，时间步长1/20=0.05s
Nt = 20
dt = 1 / Nt

# 计算时间步并预测
for n in range(0, Nt + 1):
    t = n * dt
    t_list = t * np.ones((len(x), 1))
    x_pred = np.concatenate([x[:, None], y[:, None], t_list], axis=1)
    y_pred = model.predict(x_pred)
    y_p = y_pred.flatten()
    data_n = np.concatenate([x_pred, y_pred], axis=1)
    if n == 0:
        data = data_n[:, :, None]
    else:
        data = np.concatenate([data, data_n[:, :, None]], axis=2)

# 创建图片保存路径
work_path = os.path.join('2DCD-multi',)
isCreated = os.path.exists(work_path)
if not isCreated:
    os.makedirs(work_path)
    print("保存路径: " + work_path)

# 获得y的最大值和最小值
y_min = data.min(axis=(0, 2,))[3]
y_max = data.max(axis=(0, 2,))[3]
fig = plt.figure(100, figsize=(10, 10))

def anim_update(t_id):
    plt.clf()
    # 获取当前时间步的x、y值以及预测值
    x1_t, x2_t, y_p_t = data[:, 0:1, t_id], data[:, 1:2, t_id], data[:, 3:4, t_id]
    x1_t, x2_t, y_p_t = x1_t.flatten(), x2_t.flatten(), y_p_t.flatten()

    # 只保留有效区域内的数据点
    mask_t = np.array([geom.inside(np.array([xi, yi])) for xi, yi in zip(x1_t, x2_t)])
    x1_t, x2_t, y_p_t = x1_t[mask_t], x2_t[mask_t], y_p_t[mask_t]

    plt.subplot(1, 1, 1)

    plt.scatter(x1_t, x2_t, c=y_p_t, cmap="coolwarm", s=1)

    # ax = plt.gca()
    # rect1 = patches.Rectangle([0, 0], 1.5, 1.5, linewidth=2, edgecolor='r', facecolor='none')
    # rect2 = patches.Rectangle([1, 1], 1, 1, linewidth=2, edgecolor='g', facecolor='none')
    # rect3 = patches.Rectangle([-1.5, -1.5], 2, 2, linewidth=2, edgecolor='b', facecolor='none')
    # ax.add_patch(rect1)
    # ax.add_patch(rect2)
    # ax.add_patch(rect3)

    # 使用有效区域内的数据点绘制等高线填充图
    # plt.tricontourf(x1_t, x2_t, y_p_t, levels=160, cmap="coolwarm")
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=y_min, vmax=y_max), cmap="coolwarm"), ax=plt.gca())

    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("field at t = " + str(round(t_id * dt, 2)) + " s.", fontsize=12)
    plt.savefig(work_path + '//' + 'CD_multi' + str(t_id) + '.png')

anim = FuncAnimation(fig, anim_update, frames=np.arange(0, data.shape[2]).astype(np.int64), interval=200)
anim.save(work_path + "//" + "CD-multi" + str(Nt + 1) + ".gif", writer="pillow", dpi=300)

print("Done!!!")