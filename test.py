import numpy as np
import matplotlib.pyplot as plt


X = np.arange(1, 3, 1)
Y = np.arange(0, 2, 1)
u, v = np.meshgrid(X, Y)

magnitude = np.sqrt(u**2 + v**2)
angle = np.arctan2(u, v)

# 將角度轉換為 0 到 360 的範圍
angl_degree = (angle* 180 / np.pi)

polar = np.zeros((len(Y), len(X), 2))

# 儲存 magnitude 和 angl_degree 到 polar 中
polar[:,:,0] = magnitude
polar[:,:,1] = angl_degree
# 繪製合併的位移向量場
u = u.astype("int16")
u = u - 128
v = v.astype("int16")
v = v - 128


plt.quiver(u, v, color="red", scale=0.01)  # scale 控制箭頭的大小
plt.title('Combined Displacement Vector Field')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()



C = np.sin(U)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, C)
plt.show()