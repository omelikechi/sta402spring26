# Sampling from a univariate normal distribution with unknown mean and variance

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma, multivariate_normal
from scipy.special import gammaln

np.random.seed(302)

#----------------------------------------------------------------
# Plotting and animation specifications
#----------------------------------------------------------------
show_plots = True
show_animation = False

# frame time (determines animation speed)
frame_time = 100

#----------------------------------------------------------------
# Simulate data
#----------------------------------------------------------------
n = 20  # number of samples
theta_true = 10  # true mean
sigma2_true = 3  # true variance

# generate data
y = np.random.normal(theta_true, np.sqrt(sigma2_true), size=n)

# sample mean
y_bar = np.mean(y)

#----------------------------------------------------------------
# Sampling parameters
#----------------------------------------------------------------
# number of mcmc samples
S = 10000

# proposal for metropolis algorithm ('normal' or 'uniform')
proposal_dist = 'normal'

# step size in metropolis algorithm
step_size = 0.1

# lag for reported autocorrelation
acf_lag = 10

#----------------------------------------------------------------
# Priors
#----------------------------------------------------------------
"""
- prior for mean: theta ~ N(mu_0, tau_0^2)
- prior for variance: sigma^2 ~ IG(nu_0/2, nu_0 sigma_0^2 / 2)
"""
# prior parameters
mu_0 = 0
tau0_sq = 5
nu_0 = 1
sigma0_sq = 1

#----------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------
# empirical variance
def s_n_sq(theta):
	return np.mean((y - theta) ** 2)

# autocorrelation
def lag_autocorr(x,lag):
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

#----------------------------------------------------------------
# Gibbs sampler
#----------------------------------------------------------------
def gibbs_sampler(y, S, burn=0):

	n = len(y)
	y_bar = np.mean(y)

	theta = y_bar
	sigma2 = np.var(y)

	theta_chain = np.zeros(S)
	sigma2_chain = np.zeros(S)

	for t in range(S + burn):

		# theta | sigma^2, y
		tau_n_sq = 1 / (1 / tau0_sq + n / sigma2)
		mu_n = (sigma2 * mu_0 + n * tau0_sq * y_bar) / (sigma2 + n * tau0_sq)

		theta = np.random.normal(mu_n, np.sqrt(tau_n_sq))

		# sigma^2 | theta, y
		nu_n = nu_0 + n
		sn_sq = np.mean((y - theta) ** 2)
		sigma_n_sq = (nu_0 * sigma0_sq + n * sn_sq) / nu_n

		sigma2 = invgamma.rvs(a=nu_n / 2, scale=nu_n * sigma_n_sq / 2)

		if t >= burn:

			idx = t - burn
			theta_chain[idx] = theta
			sigma2_chain[idx] = sigma2

	return theta_chain, sigma2_chain

theta_gibbs, sigma2_gibbs = gibbs_sampler(y, S)

#----------------------------------------------------------------
# Metropolis
#----------------------------------------------------------------
def log_posterior(theta, sigma2, y):

	# prevent negative values for variance
	if sigma2 <= 0:
		return -np.inf

	n = len(y)

	# likelihood
	ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum((y - theta) ** 2) / sigma2

	# prior theta
	lp_theta = -0.5 * np.log(2 * np.pi * tau0_sq) - 0.5 * (theta - mu_0) ** 2 / tau0_sq

	# prior sigma^2
	a0 = nu_0 / 2
	b0 = nu_0 * sigma0_sq / 2

	lp_sigma2 = a0 * np.log(b0) - gammaln(a0) - (a0 + 1) * np.log(sigma2) - b0 / sigma2

	return ll + lp_theta + lp_sigma2


def metropolis(y, S, proposal_dist='normal', step_size=0.1, burn=0):

	if proposal_dist not in ['normal', 'uniform']:
		raise ValueError("proposal_dist must be either 'normal' or 'uniform'")

	proposal_cov = step_size * np.eye(2)

	theta = np.mean(y)
	sigma2 = np.var(y)

	chain = np.zeros((S, 2))
	accept = 0

	for t in range(S + burn):

		if proposal_dist == 'normal':
			proposal = multivariate_normal.rvs(mean=[theta, sigma2], cov=proposal_cov)
			theta_prop, sigma2_prop = proposal
		else:
			theta_prop = theta + np.random.uniform(-step_size, step_size)
			sigma2_prop = sigma2 + np.random.uniform(-step_size, step_size)

		log_alpha = log_posterior(theta_prop, sigma2_prop, y) - log_posterior(theta, sigma2, y)

		if np.log(np.random.rand()) < log_alpha:

			theta = theta_prop
			sigma2 = sigma2_prop

			if t >= burn:
				accept += 1

		if t >= burn:
			chain[t - burn, :] = [theta, sigma2]

	theta_chain = chain[:, 0]
	sigma2_chain = chain[:, 1]

	acc_rate = accept / S

	return theta_chain, sigma2_chain, acc_rate

theta_mh, sigma2_mh, acc_rate = metropolis(y, S, proposal_dist=proposal_dist, step_size=step_size)

# print(f'Metropolis proposal distribution: {proposal_dist}')
# print(f'Metropolis acceptance rate: {acc_rate:.3f}')

#----------------------------------------------------------------
# Posterior summaries and diagnostics
#----------------------------------------------------------------
def summarize(name, theta_draws, sigma2_draws, acf_lag=10):

	print(f'\n{name}')
	print(32*f'-')
	print(f' - theta mean = {np.mean(theta_draws):.3f}')
	print(f' - sigma^2 mean = {np.mean(sigma2_draws):.3f}')
	print(f' - acf_{acf_lag}(theta) = {lag_autocorr(theta_draws, acf_lag):.3f}')
	print(f' - acf_{acf_lag}(sigma^2) = {lag_autocorr(sigma2_draws, acf_lag):.3f}')
	print(f' - ESS(theta) = {ess(theta_draws):.0f}')
	print(f' - ESS(sigma^2) = {ess(sigma2_draws):.0f}')

summarize('Gibbs sampler', theta_gibbs, sigma2_gibbs, acf_lag=acf_lag)
summarize('Metropolis', theta_mh, sigma2_mh, acf_lag=acf_lag)

#----------------------------------------------------------------
# True posterior density (grid evaluation)
#----------------------------------------------------------------
# reuse the log_posterior function defined above
# theta marginal via grid integration
theta_grid = np.linspace(min(theta_gibbs.min(), theta_mh.min()) - 1,
						 max(theta_gibbs.max(), theta_mh.max()) + 1, 400)

sigma2_grid = np.linspace(0.001,
						  max(sigma2_gibbs.max(), sigma2_mh.max()) * 1.5, 400)

# compute marginal p(theta | y)
log_joint = np.zeros((len(theta_grid), len(sigma2_grid)))

for i, th in enumerate(theta_grid):
	for j, s2 in enumerate(sigma2_grid):
		log_joint[i, j] = log_posterior(th, s2, y)

# stabilize
log_joint -= np.max(log_joint)
joint = np.exp(log_joint)

# numerical integration
p_theta = np.trapezoid(joint, sigma2_grid, axis=1)
p_sigma2 = np.trapezoid(joint, theta_grid, axis=0)

# normalize
p_theta /= np.trapezoid(p_theta, theta_grid)
p_sigma2 /= np.trapezoid(p_sigma2, sigma2_grid)

#----------------------------------------------------------------
# Plots
#----------------------------------------------------------------
if show_plots:
	fig, axes = plt.subplots(2, 3, figsize=(18, 9))

	# theta trace plot (gibbs)
	axes[0, 0].plot(theta_gibbs, alpha=0.7)
	axes[0, 0].set_title('Gibbs trace: theta')

	# theta trace plot (metropolis)
	axes[0, 1].plot(theta_mh, alpha=0.7)
	axes[0, 1].set_title(f'Metropolis trace: theta ({proposal_dist})')

	# sigma^2 trace plot (gibbs)
	axes[1, 0].plot(sigma2_gibbs, alpha=0.7)
	axes[1, 0].set_title('Gibbs trace: sigma^2')

	# sigma^2 trace plot (metropolis)
	axes[1, 1].plot(sigma2_mh, alpha=0.7)
	axes[1, 1].set_title(f'Metropolis trace: sigma^2 ({proposal_dist})')

	# theta posterior
	axes[0, 2].hist(theta_gibbs, bins=40, density=True, alpha=0.5, label='Gibbs')
	axes[0, 2].hist(theta_mh, bins=40, density=True, alpha=0.5, label='Metropolis')
	axes[0, 2].plot(theta_grid, p_theta, 'k', linewidth=2, label='True density')
	axes[0, 2].set_title('Posterior of theta')
	axes[0, 2].legend()

	# sigma^2 posterior
	axes[1, 2].hist(sigma2_gibbs, bins=40, density=True, alpha=0.5, label='Gibbs')
	axes[1, 2].hist(sigma2_mh, bins=40, density=True, alpha=0.5, label='Metropolis')
	axes[1, 2].plot(sigma2_grid, p_sigma2, 'k', linewidth=2, label='True density')
	axes[1, 2].set_title('Posterior of sigma^2')
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

	fig, axes = plt.subplots(2, 2, figsize=(18, 9))

	trace_theta_g, = axes[0, 0].plot([], [])
	trace_theta_m, = axes[0, 1].plot([], [])

	hist_theta_ax = axes[1, 0]
	hist_sigma_ax = axes[1, 1]

	axes[0, 0].set_title('Gibbs trace: theta')
	axes[0, 1].set_title(f'Metropolis trace: theta ({proposal_dist})')
	axes[1, 0].set_title('Posterior theta')
	axes[1, 1].set_title('Posterior sigma^2')

	axes[0, 0].set_xlim(0, S)
	axes[0, 1].set_xlim(0, S)

	axes[0, 0].set_ylim(theta_gibbs.min(), theta_gibbs.max())
	axes[0, 1].set_ylim(theta_mh.min(), theta_mh.max())

	def update(frame):

		k = frame * step

		trace_theta_g.set_data(np.arange(k), theta_gibbs[:k])
		trace_theta_m.set_data(np.arange(k), theta_mh[:k])

		hist_theta_ax.cla()
		hist_sigma_ax.cla()

		hist_theta_ax.hist(theta_gibbs[:k], bins=40, density=True, alpha=0.5, label='Gibbs')
		hist_theta_ax.hist(theta_mh[:k], bins=40, density=True, alpha=0.5, label='Metropolis')
		hist_theta_ax.plot(theta_grid, p_theta, 'k', linewidth=2, label='True density')
		hist_theta_ax.set_title('Posterior of theta')
		hist_theta_ax.set_xlabel('theta')
		hist_theta_ax.set_ylabel('density')
		hist_theta_ax.legend()

		hist_sigma_ax.hist(sigma2_gibbs[:k], bins=40, density=True, alpha=0.5, label='Gibbs')
		hist_sigma_ax.hist(sigma2_mh[:k], bins=40, density=True, alpha=0.5, label='Metropolis')
		hist_sigma_ax.plot(sigma2_grid, p_sigma2, 'k', linewidth=2, label='True density')
		hist_sigma_ax.set_title('Posterior of sigma^2')
		hist_sigma_ax.set_xlabel('sigma^2')
		hist_sigma_ax.set_ylabel('density')
		hist_sigma_ax.legend()

		return trace_theta_g, trace_theta_m

	anim = FuncAnimation(fig, update, frames=frames, interval=frame_time)

	plt.show()


