import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# 定义问题的几何区域
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 扩散系数
nu = 1e-3  # 粘性系数
v_x = 0.5  # 对流速度的 x 方向分量
v_y = 0  # 对流速度的 y 方向分量

# 定义 PDE（扩散对流方程）
def pde(t, c):

    c_t = dde.grad.jacobian(c, t, i=0, j=2)  # ∂u/∂t

    # 一阶导数（对流项）
    c_x = dde.grad.jacobian(c, t, i=0, j=0)  # ∂u/∂x
    c_y = dde.grad.jacobian(c, t, i=0, j=1)  # ∂u/∂y

    # 二阶导数（扩散项）
    c_xx = dde.grad.hessian(c, t, component=0, i=0, j=0)  # ∂²u/∂x²
    c_yy = dde.grad.hessian(c, t, component=0, i=1, j=1)  # ∂²u/∂y²

    convection = v_x * c_x + v_y * c_y  # 对流项
    diffusion = nu * (c_xx + c_yy)  # 扩散项

    # PDE 方程
    pde_equation = c_t + diffusion - convection

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


bc_t = dde.icbc.NeumannBC(geomtime, lambda x: 2, boundary_t)
bc_b = dde.icbc.NeumannBC(geomtime, lambda x: 2, boundary_b)
bc_l = dde.icbc.NeumannBC(geomtime, lambda x: 2, boundary_l)
bc_r = dde.icbc.NeumannBC(geomtime, lambda x: 2, boundary_r)

# 初始条件：一个扩散源
def initial_condition(c):
    return 0.1 * np.ones_like(c)[:, 0]

ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial,)

# 设置数据（PDE 方程和边界条件）
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

losshistory, train_state = model.train(iterations=10000, display_every=1000, callbacks=[early_stopping])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# x, y方向离散200个节点
x1 = np.linspace(0, 1, num=200, endpoint=True).flatten()
y1 = np.linspace(0, 1, num=200, endpoint=True).flatten()
xx1, yy1 = np.meshgrid(x1, y1)
x = xx1.flatten()
y = yy1.flatten()

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

    print(x_pred.shape, y_pred.shape)
    print(data.shape, data_n.shape)

# 创建图片保存路径
work_path = os.path.join('2DCD',)
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
    x1_t, x2_t, y_p_t = data[:, 0:1, t_id], data[:, 1:2, t_id], data[:, 3:4, t_id]
    x1_t, x2_t, y_p_t = x1_t.flatten(), x2_t.flatten(), y_p_t.flatten()
    print(t_id, x1_t.shape, x1_t.shape, y_p_t.shape)

    plt.subplot(1, 1, 1)
    plt.tricontourf(x1_t, x2_t, y_p_t, levels=160, cmap="coolwarm")
    cb0 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=y_min, vmax=y_max), cmap="coolwarm"), ax=plt.gca())

    plt.xlabel('$x (m)$')
    plt.ylabel('$y (m)$')
    plt.title("field at t = " + str(round(t_id * dt, 2)) + " s.", fontsize=12)
    plt.savefig(work_path + '//' + 'CD_' + str(t_id) + '.png')

    print("data.shape[2] = ", data.shape[2])

anim = FuncAnimation(fig, anim_update, frames=np.arange(0, data.shape[2]).astype(np.int64), interval=200)
anim.save(work_path + "//" + "CD-" + str(Nt + 1) + ".gif", writer="pillow", dpi=300)
