import numpy as np
import matplotlib.pyplot as plt


X = np.arange(1, 3, 1)
Y = np.arange(0, 2, 1)
u, v = np.meshgrid(X, Y)

magnitude = np.sqrt(u**2 + v**2)
angle = np.arctan2(u, v)

# �N�����ഫ�� 0 �� 360 ���d��
angl_degree = (angle* 180 / np.pi)

polar = np.zeros((len(Y), len(X), 2))

# �x�s magnitude �M angl_degree �� polar ��
polar[:,:,0] = magnitude
polar[:,:,1] = angl_degree
# ø�s�X�֪��첾�V�q��
u = u.astype("int16")
u = u - 128
v = v.astype("int16")
v = v - 128


plt.quiver(u, v, color="red", scale=0.01)  # scale ����b�Y���j�p
plt.title('Combined Displacement Vector Field')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()



C = np.sin(U)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, C)
plt.show()