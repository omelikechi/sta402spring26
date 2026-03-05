import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Continuous-state Markov chain on R: AR(1) / discretized OU
# X_{t+1} = a X_t + sigma * Z_t, Z_t ~ N(0,1)
rng = np.random.default_rng(302)

a = 1/np.sqrt(2)
sigma = 1
T = 25
interval = 500

N = 100000

# Stationary distribution (for |a| < 1): N(0, sigma^2 / (1 - a^2))
stat_var = sigma**2 / (1.0 - a**2)
stat_sd = np.sqrt(stat_var)

# initial state
# x = -4 * np.ones(N)

z = np.random.choice([0,1], size=N, p=[0.25,0.75])
x = np.where(z==0, np.random.normal(-4, 1, N), np.random.normal(4, 1, N))

# Plot setup
fig, ax = plt.subplots(figsize=(14,7))
bins = 80
xlim = (-5 * stat_sd, 5 * stat_sd)

hist = ax.hist(x, bins=bins, range=xlim, color='deepskyblue', density=True, alpha=0.7)[2]
grid = np.linspace(xlim[0], xlim[1], 600)

# stationary pdf overlay
stat_pdf = (1.0 / (stat_sd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (grid / stat_sd) ** 2)
(pdf_line,) = ax.plot(grid, stat_pdf, linewidth=2)

ax.set_xlim(*xlim)
ax.set_ylim(0, stat_pdf.max() * 1.25)
ax.set_xlabel(r'$\theta$', fontsize=24)
ax.set_ylabel('Density', fontsize=24)
title = ax.set_title('t = 0', fontsize=24)

def step(x):
	return a * x + sigma * rng.standard_normal(x.shape[0])

def update(t):
	global x, hist
	if t > 0:
		x = step(x)

	# clear old bars
	for rect in hist:
		rect.remove()

	hist = ax.hist(x, bins=bins, range=xlim, color='deepskyblue', density=True, alpha=0.7)[2]
	title.set_text(f't = {t}')
	return list(hist) + [pdf_line, title]

ani = FuncAnimation(fig, update, frames=T + 1, interval=interval, repeat=False)

plt.show()


