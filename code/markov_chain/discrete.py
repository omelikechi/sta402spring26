import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Column-stochastic transition kernel on {1,2,3}
# Columns sum to 1
K = np.array([
	[0.6, 0.2, 0.1],
	[0.3, 0.6, 0.3],
	[0.1, 0.2, 0.6],
])

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(K)

# eigenvalues
print(f'\nEigenvalues: {eigvals}\n')

# invariant probability measure is the eigenvector that corresponds to eigenvalue 1
pi = eigvecs[:,0] / np.sum(eigvecs[:,0])
print(f'Invariant distribution: {np.round(pi,3)}')

# Initial distribution (column vector)
x = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)

T = 25

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar([1,2,3], x[:, 0])

ax.set_ylim(0, 1)
ax.set_xticks([1,2,3])
ax.set_xlabel('State', fontsize=20)
ax.set_ylabel('Probability', fontsize=20)
title = ax.set_title('t = 0', fontsize=20)

def update(t):
	global x
	if t > 0:
		x = K @ x
	for i, bar in enumerate(bars):
		bar.set_height(x[i, 0])
	title.set_text(f't = {t}')
	return bars

ani = FuncAnimation(fig, update, frames=T + 1, interval=1000, repeat=False)

plt.show()


