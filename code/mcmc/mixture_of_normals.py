# Sampling from a mixture of normal densities

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(302)

#----------------------------------------------------------------
# Plotting and animation specifications
#----------------------------------------------------------------
show_plots = True
show_animation = False

# frame time (determines animation speed)
frame_time = 100

#----------------------------------------------------------------
# Target density: mixture of two normal densities
#----------------------------------------------------------------
"""
Target density is a mixture of normal distributions:
	pi(x) = w * N(mu_1, sigma_1^2) + (1 - w) * N(mu_2, sigma_2^2)
"""

# mixture weight on the first component
w = 0.5

# component means
mu_1 = -6
mu_2 = 6

# component standard deviations
sigma_1 = 1
sigma_2 = 1

#----------------------------------------------------------------
# Sampling parameters
#----------------------------------------------------------------
# number of mcmc samples per chain
S = 10000

# proposal for metropolis algorithm ('normal' or 'uniform')
proposal_dist = 'normal'

# step size in metropolis algorithm
step_size = 0.5

# lag for reported autocorrelation
acf_lag = 10

# number of burn-in draws
burn = 0

# starting points for separate chains
x0_left = mu_1
x0_right = mu_2
x0_center = (mu_1 + mu_2) / 2

#----------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------
# autocorrelation
def lag_autocorr(x, lag):
	x = np.asarray(x)
	n = len(x)
	x_mean = np.mean(x)
	c0 = np.sum((x - x_mean) ** 2) / n
	c_lag = np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / (n - lag)
	return c_lag / c0

# essential sample size
def ess(x):
	x = np.asarray(x)
	x = x.reshape((1, -1))
	return float(az.ess(x))

# mixture of normal densities
def mixture_density(x):
	return w * norm.pdf(x, loc=mu_1, scale=sigma_1) + (1 - w) * norm.pdf(x, loc=mu_2, scale=sigma_2)

# use logs only after summing the mixture on the density scale
def log_target(x):
	dens = mixture_density(x)
	return np.log(dens)

#----------------------------------------------------------------
# Metropolis
#----------------------------------------------------------------
def metropolis(S, x0, proposal_dist='normal', step_size=0.1, burn=0):

	x = x0

	chain = np.zeros(S)
	accept = 0

	for t in range(S + burn):

		# propose a local move
		if proposal_dist == 'normal':
			x_prop = x + np.random.normal(0, np.sqrt(step_size))
		else:
			x_prop = x + np.random.uniform(-step_size, step_size)

		# because the proposal is symmetric, the proposal terms cancel
		log_alpha = log_target(x_prop) - log_target(x)

		if np.log(np.random.rand()) < log_alpha:
			x = x_prop

			if t >= burn:
				accept += 1

		if t >= burn:
			chain[t - burn] = x

	acc_rate = accept / S

	return chain, acc_rate

# run chains from different initial states
chain_left, acc_rate_left = metropolis(S, x0=x0_left, proposal_dist=proposal_dist, step_size=step_size, burn=burn)
chain_right, acc_rate_right = metropolis(S, x0=x0_right, proposal_dist=proposal_dist, step_size=step_size, burn=burn)
chain_center, acc_rate_center = metropolis(S, x0=x0_center, proposal_dist=proposal_dist, step_size=step_size, burn=burn)

#----------------------------------------------------------------
# Diagnostics and summaries
#----------------------------------------------------------------
def summarize(name, draws, acf_lag=10):

	print(f'\n{name}')
	print(32 * '-')
	print(f' - mean = {np.mean(draws):.3f}')
	print(f' - sd = {np.std(draws):.3f}')
	print(f' - acf_{acf_lag} = {lag_autocorr(draws, acf_lag):.3f}')
	print(f' - ESS = {ess(draws):.0f}')

# print(f'proposal distribution = {proposal_dist}')
# print(f'step size = {step_size}')
# print(f'acceptance rate (left start) = {acc_rate_left:.3f}')
# print(f'acceptance rate (right start) = {acc_rate_right:.3f}')
# print(f'acceptance rate (center start) = {acc_rate_center:.3f}')

summarize(f'chain starting at {x0_left}', chain_left, acf_lag=acf_lag)
summarize(f'chain starting at {x0_right}', chain_right, acf_lag=acf_lag)
summarize(f'chain starting at {x0_center}', chain_center, acf_lag=acf_lag)

#----------------------------------------------------------------
# Grid evaluation of the true target density
#----------------------------------------------------------------
# include enough range to see both modes and any rare jumps between them
x_min = min(chain_left.min(), chain_right.min(), chain_center.min(), mu_1 - 4 * sigma_1, mu_2 - 4 * sigma_2) - 1
x_max = max(chain_left.max(), chain_right.max(), chain_center.max(), mu_1 + 4 * sigma_1, mu_2 + 4 * sigma_2) + 1

x_grid = np.linspace(x_min, x_max, 1000)
p_grid = mixture_density(x_grid)

#----------------------------------------------------------------
# Plots
#----------------------------------------------------------------
if show_plots:
	fig, axes = plt.subplots(2, 3, figsize=(18,9), sharey='row')

	# trace plot for chain started near the left mode
	axes[0, 0].plot(chain_left, alpha=0.8)
	axes[0, 0].axhline(mu_1, linestyle='--')
	axes[0, 0].axhline(mu_2, linestyle='--')
	axes[0, 0].set_title(f'trace plot: start at {x0_left}')
	axes[0, 0].set_xlabel('iteration')
	axes[0, 0].set_ylabel('x')

	# trace plot for chain started near the right mode
	axes[0, 1].plot(chain_right, alpha=0.8)
	axes[0, 1].axhline(mu_1, linestyle='--')
	axes[0, 1].axhline(mu_2, linestyle='--')
	axes[0, 1].set_title(f'trace plot: start at {x0_right}')
	axes[0, 1].set_xlabel('iteration')
	axes[0, 1].set_ylabel('x')

	# trace plot for chain started between the modes
	axes[0, 2].plot(chain_center, alpha=0.8)
	axes[0, 2].axhline(mu_1, linestyle='--')
	axes[0, 2].axhline(mu_2, linestyle='--')
	axes[0, 2].set_title(f'trace plot: start at {x0_center}')
	axes[0, 2].set_xlabel('iteration')
	axes[0, 2].set_ylabel('x')

	# histogram for chain started near the left mode
	axes[1, 0].hist(chain_left, bins=60, density=True, alpha=0.6, label='MCMC draws')
	axes[1, 0].plot(x_grid, p_grid, 'k', linewidth=2, label='target density')
	axes[1, 0].set_title('histogram: start near left mode')
	axes[1, 0].set_xlabel('x')
	axes[1, 0].set_ylabel('density')
	axes[1, 0].legend()

	# histogram for chain started near the right mode
	axes[1, 1].hist(chain_right, bins=60, density=True, alpha=0.6, label='MCMC draws')
	axes[1, 1].plot(x_grid, p_grid, 'k', linewidth=2, label='target density')
	axes[1, 1].set_title('histogram: start near right mode')
	axes[1, 1].set_xlabel('x')
	axes[1, 1].set_ylabel('density')
	axes[1, 1].legend()

	# histogram overlay for all three chains
	axes[1, 2].hist(chain_left, bins=60, density=True, alpha=0.35, label='left start')
	axes[1, 2].hist(chain_right, bins=60, density=True, alpha=0.35, label='right start')
	axes[1, 2].hist(chain_center, bins=60, density=True, alpha=0.35, label='center start')
	axes[1, 2].plot(x_grid, p_grid, 'k', linewidth=2, label='target density')
	axes[1, 2].set_title('histograms from different initializations')
	axes[1, 2].set_xlabel('x')
	axes[1, 2].set_ylabel('density')
	axes[1, 2].legend()

	plt.tight_layout()
	plt.show()

#----------------------------------------------------------------
# Animation of MCMC evolution
#----------------------------------------------------------------
if show_animation:
	from matplotlib.animation import FuncAnimation

	step = 100
	frames = S // step

	fig, axes = plt.subplots(2, 2, figsize=(16, 10))

	trace_left, = axes[0, 0].plot([], [])
	trace_right, = axes[0, 1].plot([], [])

	hist_left_ax = axes[1, 0]
	hist_right_ax = axes[1, 1]

	axes[0, 0].set_title('trace: start near left mode')
	axes[0, 1].set_title('trace: start near right mode')
	axes[1, 0].set_title('histogram: start near left mode')
	axes[1, 1].set_title('histogram: start near right mode')

	axes[0, 0].set_xlim(0, S)
	axes[0, 1].set_xlim(0, S)
	axes[0, 0].set_ylim(x_min, x_max)
	axes[0, 1].set_ylim(x_min, x_max)

	def update(frame):

		k = frame * step

		trace_left.set_data(np.arange(k), chain_left[:k])
		trace_right.set_data(np.arange(k), chain_right[:k])

		hist_left_ax.cla()
		hist_right_ax.cla()

		hist_left_ax.hist(chain_left[:k], bins=60, density=True, alpha=0.6, label='left start')
		hist_left_ax.plot(x_grid, p_grid, 'k', linewidth=2, label='target density')
		hist_left_ax.set_title('histogram: start near left mode')
		hist_left_ax.set_xlabel('x')
		hist_left_ax.set_ylabel('density')
		hist_left_ax.legend()

		hist_right_ax.hist(chain_right[:k], bins=60, density=True, alpha=0.6, label='right start')
		hist_right_ax.plot(x_grid, p_grid, 'k', linewidth=2, label='target density')
		hist_right_ax.set_title('histogram: start near right mode')
		hist_right_ax.set_xlabel('x')
		hist_right_ax.set_ylabel('density')
		hist_right_ax.legend()

		return trace_left, trace_right

	anim = FuncAnimation(fig, update, frames=frames, interval=frame_time)

	plt.tight_layout()
	plt.show()

print()
