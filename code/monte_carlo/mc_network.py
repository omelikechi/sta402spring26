# Monte Carlo problem taken from https://math.arizona.edu/~tgk/mc/hmwk1.pdf

import matplotlib.pyplot as plt
import numpy as np

sample_size = 1000

# t1 = np.random.uniform(0, 1, sample_size)
# t2 = np.random.uniform(0, 2, sample_size)
# t3 = np.random.uniform(0, 3, sample_size)
# t4 = np.random.uniform(0, 1, sample_size)
# t5 = np.random.uniform(0, 2, sample_size)

# min_time = np.minimum.reduce([t1 + t4, t1 + t3 + t5, t2 + t5, t2 + t3 + t4])

# mean = np.mean(min_time)
# var = np.var(min_time)
# std = np.sqrt(var)

# print(f'mean = {mean:.4f}')
# print(f'var = {var:.4f}')
# print(f'std = {std:.4f}')
# print(f'standard error = {std / mean:.4f}')

n_trajectories = 1000
barS = []
barX = []

fig, ax = plt.subplots(1, 2, figsize=(18,6))

for n in range(n_trajectories):
	X = np.random.normal(0, 1, sample_size)
	S = np.concatenate(([0], np.cumsum(X)))[:-1]
	barX.append(np.sqrt(sample_size) * np.mean(X))
	barS.append(np.sqrt(sample_size) * S[-1])

	# ax[0].plot(S)

ax[0].hist(barS, edgecolor='black', density=True, bins=20)
ax[1].hist(barX, edgecolor='black', density=True, bins=20)



plt.tight_layout()
plt.show()



