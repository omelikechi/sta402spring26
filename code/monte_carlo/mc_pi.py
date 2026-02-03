# Estimate the volume of unit ball of radius 1 using Monte Carlo

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

#----------------------------------------------------------------
# Setup
#----------------------------------------------------------------

np.random.seed(302)

# number of samples
S = 10000

# scaling factor
scale = 1 / np.sqrt(S)

# number of trajectories
n_trajectories = 100

# circle
plot_circle = False

# trajectories
plot_trajectories = False
plot_envelope = False

# histograms
plot_histograms = False
scale_historgrams = False

# higher dimensions
d = 5
higher_dimensions = True
plot_dim_vs_vol = False

#----------------------------------------------------------------
# Plot samples
#----------------------------------------------------------------
if plot_circle:

	x = np.random.uniform(-1, 1, size=S)
	y = np.random.uniform(-1, 1, size=S)
	samples_inside = (x**2 + y**2) < 1

	plt.figure(figsize=(10,9))

	# points
	point_size = 50 if S < 1000 else 10
	plt.scatter(x[samples_inside], y[samples_inside], s=point_size, color='dodgerblue')
	plt.scatter(x[~samples_inside], y[~samples_inside], s=point_size, color='orange')

	# unit circle boundary
	theta = np.linspace(0, 2*np.pi, 400)
	plt.plot(np.cos(theta), np.sin(theta), color='black')

	# axes lines
	plt.axvline(0, color='black')
	plt.axhline(0, color='black')

	plt.gca().set_aspect('equal')
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.tight_layout()
	plt.show()

#----------------------------------------------------------------
# Plot trajectories
#----------------------------------------------------------------
if plot_trajectories:

	plt.figure(figsize=(16,8))

	for _ in range(n_trajectories):

		x = np.random.uniform(-1, 1, size=S)
		y = np.random.uniform(-1, 1, size=S)
		samples_inside = (x**2 + y**2) < 1

		# cumulative π estimate path (length S)
		pi_est = 4 * np.cumsum(samples_inside) / np.arange(1, S+1)

		alpha = 1 if n_trajectories <= 5 else 0.5
		plt.plot(pi_est, alpha=alpha)

	n = np.arange(1, S+1)
	convergence_rate = 1 / np.sqrt(n)

	# plot convergence envelope
	if plot_envelope:
		upper = np.pi + convergence_rate
		lower = np.pi - convergence_rate
		plt.plot(upper, color='black', linestyle=':', lw=3)
		plt.plot(lower, color='black', linestyle=':', lw=3)

	plt.axhline(np.pi, color='black', linestyle='--', lw=3)
	plt.ylim(np.pi - 10 * scale, np.pi + 10 * scale)	

	plt.xlabel(r'Number of samples, $S$', fontsize=22)
	plt.title(r'$\hat{\pi} = \frac{4}{S}\sum_{i=1}^S \mathbb{1}(x_i^2 + y_i^2 < 1)$', fontsize=22)
	plt.tight_layout()
	plt.show()

#----------------------------------------------------------------
# Plot histograms
#----------------------------------------------------------------
if plot_histograms:

	# sample sizes on log scale
	sizes = np.unique(np.logspace(np.log10(10), np.log10(S), 6).astype(int))

	sharex = False if scale_historgrams else True
	fig, axes = plt.subplots(2, 3, figsize=(18,8), sharex=sharex)
	axes = axes.ravel()

	estimates = np.empty((n_trajectories, len(sizes)))

	for r in range(n_trajectories):
		x = np.random.uniform(-1, 1, size=S)
		y = np.random.uniform(-1, 1, size=S)
		inside = (x**2 + y**2) < 1

		for j, s in enumerate(sizes):
			estimates[r, j] = 4 * inside[:s].mean()

	for j, (s, ax) in enumerate(zip(sizes, axes)):
		if scale_historgrams:
			ax.hist(np.sqrt(s) * (estimates[:,j] - np.pi), bins=25, color='deepskyblue', edgecolor='black', density=True)
			ax.axvline(0, color='black', linestyle='--', lw=3)
		else:
			ax.hist(estimates[:,j], bins=25, color='deepskyblue', edgecolor='black', density=True)
			ax.axvline(np.pi, color='black', linestyle='--', lw=3)
		ax.set_title(f'S = {s}', fontsize=18)

	plt.tight_layout()
	plt.show()

#----------------------------------------------------------------
# Higher dimensions
#----------------------------------------------------------------
if higher_dimensions:

	# true unit ball volume
	def vol_unit_ball(d):
		return np.pi**(d/2) / gamma(d/2 + 1)

	if plot_dim_vs_vol:

		plt.figure(figsize=(16,8))
		dims = np.arange(1, 30 + 1)
		plt.plot(dims, vol_unit_ball(dims))
		plt.xlabel('Dimension', fontsize=18)
		plt.ylabel('Volume', fontsize=18)
		plt.title(f'Volume of unit ball in different dimensions', fontsize=24)
		plt.tight_layout()
		plt.show()

	true_volume = vol_unit_ball(d)

	plt.figure(figsize=(16,8))

	for _ in range(n_trajectories):

		x = np.random.uniform(-1, 1, size=(S, d))
		samples_inside = (x**2).sum(axis=1) < 1

		vol_est = 2**d * np.cumsum(samples_inside) / np.arange(1, S+1)

		alpha = 1 if n_trajectories <= 5 else 0.5
		plt.plot(vol_est, alpha=alpha)

	n = np.arange(1, S+1)
	convergence_rate = 1 / np.sqrt(n)

	if plot_envelope:
		upper = true_volume + convergence_rate
		lower = true_volume - convergence_rate
		plt.plot(upper, color='black', linestyle=':', lw=3)
		plt.plot(lower, color='black', linestyle=':', lw=3)

	plt.axhline(true_volume, color='black', linestyle='--', lw=3)
	plt.ylim(true_volume - d * 20 * scale, true_volume + d * 20 * scale)	

	plt.xlabel(r'Number of samples, $S$', fontsize=22)
	plt.title(f'Monte Carlo estimate of unit ball volume in $\\mathbb{{R}}^{{{d}}}$', fontsize=22)
	plt.tight_layout()
	plt.show()



