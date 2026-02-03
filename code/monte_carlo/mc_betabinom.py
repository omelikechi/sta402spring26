# Monte Carlo for beta-binomial model

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

#----------------------------------------------------------------
# data and prior
#----------------------------------------------------------------
# beta prior parameters
a = 1
b = 1

# number of observed samples (bernoulli trials)
n = 10

# number of successes
y = 3

# number of monte carlo samples
S = 1000

# number of monte carlo trajectories
n_trajectories = 1

#----------------------------------------------------------------
# posterior parameters: beta(a + y, b + n - y)
#----------------------------------------------------------------
a_post = a + y
b_post = b + n - y

#----------------------------------------------------------------
# analytic posterior quantities (true values)
#----------------------------------------------------------------
# posterior mean
post_mean = a_post / (a_post + b_post)

# posterior variance
post_var = (a_post * b_post) / ((a_post + b_post)**2 * (a_post + b_post + 1))

# probability theta < 0.75
prob_less_075 = beta.cdf(0.75, a_post, b_post)

# 0.1 quantile
quantile = beta.ppf(0.1, a_post, b_post)

# index for running estimates
idx = np.arange(1, S + 1)

# grid for plotting posterior density
theta_grid = np.linspace(0, 1, 500)
posterior_density = beta.pdf(theta_grid, a_post, b_post)

#----------------------------------------------------------------
# plotting multiple MC trajectories + posterior distribution
#----------------------------------------------------------------

figsize = (16,6)

# (i) posterior mean
fig, axes = plt.subplots(1, 2, figsize=figsize)
for traj in range(n_trajectories):
	np.random.seed(traj)
	theta_samples = np.random.beta(a_post, b_post, size=S)
	mc_mean_path = np.cumsum(theta_samples) / idx
	axes[0].plot(idx, mc_mean_path, alpha=0.6)
axes[0].axhline(post_mean, linestyle='--', color='black', lw=2, label='true value')
axes[0].set_title('posterior mean')
axes[0].set_xlabel('number of samples')
axes[0].legend()

axes[1].plot(theta_grid, posterior_density)
axes[1].axvline(post_mean, linestyle='--', color='black', lw=2)
axes[1].set_title('posterior distribution')
axes[1].set_xlabel('theta')

plt.tight_layout()
plt.show()

# (ii) posterior variance
fig, axes = plt.subplots(1, 2, figsize=figsize)
for traj in range(n_trajectories):
	np.random.seed(traj)
	theta_samples = np.random.beta(a_post, b_post, size=S)
	mc_mean_path = np.cumsum(theta_samples) / idx
	mc_var_path = (np.cumsum(theta_samples**2) / idx) - mc_mean_path**2
	axes[0].plot(idx, mc_var_path, alpha=0.6)
axes[0].axhline(post_var, linestyle='--', color='black', lw=2, label='true value')
axes[0].set_title('posterior variance')
axes[0].set_xlabel('number of samples')
axes[0].legend()

axes[1].plot(theta_grid, posterior_density)
axes[1].set_title('posterior distribution')
axes[1].set_xlabel('theta')

plt.tight_layout()
plt.show()

# (iii) P(theta < 0.75 | y)
fig, axes = plt.subplots(1, 2, figsize=figsize)
for traj in range(n_trajectories):
	np.random.seed(traj)
	theta_samples = np.random.beta(a_post, b_post, size=S)
	mc_prob_path = np.cumsum(theta_samples < 0.75) / idx
	axes[0].plot(idx, mc_prob_path, alpha=0.6)
axes[0].axhline(prob_less_075, linestyle='--', color='black', lw=2, label='true value')
axes[0].set_title('P(theta < 0.75 | y)')
axes[0].set_xlabel('number of samples')
axes[0].legend()

axes[1].plot(theta_grid, posterior_density)
axes[1].axvline(0.75, linestyle=':', color='red')
axes[1].set_title('posterior distribution')
axes[1].set_xlabel('theta')

plt.tight_layout()
plt.show()

# (iv) 0.1 quantile
fig, axes = plt.subplots(1, 2, figsize=figsize)
for traj in range(n_trajectories):
	np.random.seed(traj)
	theta_samples = np.random.beta(a_post, b_post, size=S)
	mc_quantile_path = np.array([np.quantile(theta_samples[:i], 0.1) for i in idx])
	axes[0].plot(idx, mc_quantile_path, alpha=0.6)
axes[0].axhline(quantile, linestyle='--', color='black', lw=2, label='true value')
axes[0].set_title('0.1 posterior quantile')
axes[0].set_xlabel('number of samples')
axes[0].legend()

axes[1].plot(theta_grid, posterior_density)
axes[1].axvline(quantile, linestyle='--', color='black', lw=2)
axes[1].set_title('posterior distribution')
axes[1].set_xlabel('theta')

plt.tight_layout()
plt.show()