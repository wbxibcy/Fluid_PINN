import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# 定义矩形区域
geom_rect1 = dde.geometry.Rectangle([0, 0], [1.5, 1.5])
geom_rect2 = dde.geometry.Rectangle([1, 1], [2, 2])
geom_rect3 = dde.geometry.Rectangle([-1.5, -1.5], [0.5, 0.5])

geom = dde.geometry.CSGUnion(dde.geometry.CSGUnion(geom_rect1, geom_rect2), geom_rect3)

# 绘制几何区域，观察它的形状
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
points = np.vstack((X.flatten(), Y.flatten())).T

# 检查哪些点在复合区域内
inside = geom.inside(points)

# 绘制复合几何区域
plt.figure(figsize=(6, 6))
plt.scatter(points[inside, 0], points[inside, 1], color="blue", s=1)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Composite Geometry Region using CSGUnion')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
