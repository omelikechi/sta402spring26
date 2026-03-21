# Reading comprehension example (Hoff, Ch 7: Multivariate normal)
"""
Data: n = 22 students, each with two exam scores:
	y_i = (score before instruction, score after instruction)

Likelihood:
	y_i | theta, Sigma ~ N_2(theta, Sigma)

Priors:
	theta ~ N_2(mu0, Lambda_n)
	Sigma ~ inverse-wishart(nu0, S0)

we use sufficient statistics instead of raw data
"""

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Sufficient statistics
#----------------------------------------------------------------
n = 22
ybar = np.array([47.18, 53.86])
s1_sq = 182.16
s2_sq = 243.65
corr = 0.70
cov12 = corr * np.sqrt(s1_sq * s2_sq)

s = np.array([
	[s1_sq, cov12],
	[cov12, s2_sq]
])

#----------------------------------------------------------------
# Priors
#----------------------------------------------------------------
mu0 = np.array([50.0, 50.0])

L0 = np.array([
	[625.0, 312.5],
	[312.5, 625.0]
])

nu0 = 4
S0 = np.array([
	[625.0, 312.5],
	[312.5, 625.0]
])

#----------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------
def rmvnorm(mean, cov):
	return np.random.multivariate_normal(mean, cov)


def rwish(df, scale):
	# wishart via gaussian draws
	p = scale.shape[0]
	z = np.random.multivariate_normal(np.zeros(p), scale, size=df)
	return z.T @ z

#----------------------------------------------------------------
# Gibbs sampler
#----------------------------------------------------------------
def run_gibbs(n_iter=5000, seed=1):
	np.random.seed(seed)

	Sigma = s.copy()

	theta_samples = np.zeros((n_iter, 2))
	sigma_samples = np.zeros((n_iter, 4))

	L0_inv = np.linalg.inv(L0)

	for it in range(n_iter):
		# update theta given current Sigma
		Sigma_inv = np.linalg.inv(Sigma)
		Ln = np.linalg.inv(L0_inv + n * Sigma_inv)
		mun = Ln @ (L0_inv @ mu0 + n * Sigma_inv @ ybar)
		theta = rmvnorm(mun, Ln)

		# update Sigma given theta
		centered_mean = ybar - theta
		Sn = S0 + (n - 1) * s + n * np.outer(centered_mean, centered_mean)
		Sigma = np.linalg.inv(rwish(nu0 + n, np.linalg.inv(Sn)))

		theta_samples[it, :] = theta
		sigma_samples[it, :] = Sigma.reshape(-1)

	return theta_samples, sigma_samples

#----------------------------------------------------------------
# Run gibbs
#----------------------------------------------------------------
theta_samples, sigma_samples = run_gibbs()

#----------------------------------------------------------------
# Posterior summaries
#----------------------------------------------------------------
improvement = theta_samples[:, 1] - theta_samples[:, 0]

quantiles = np.quantile(improvement, [0.025, 0.5, 0.975])
prob_positive = np.mean(theta_samples[:, 1] > theta_samples[:, 0])

print('posterior quantiles for theta_2 - theta_1:')
print(f'2.5%   = {quantiles[0]:.6f}')
print(f'50%    = {quantiles[1]:.6f}')
print(f'97.5%  = {quantiles[2]:.6f}')
print()
print(f'posterior probability that theta_2 > theta_1: {prob_positive:.6f}')

# posterior means
# theta mean
theta_mean = theta_samples.mean(axis=0)
print()
print('posterior mean of theta:')
print(theta_mean)

# Sigma mean (reshape samples first)
sigma_mats = sigma_samples.reshape(-1, 2, 2)
Sigma_mean = sigma_mats.mean(axis=0)
print()
print('posterior mean of Sigma:')
print(Sigma_mean)

# posterior correlation (derived from covariance)
# corr = Sigma_12 / sqrt(Sigma_11 * Sigma_22)
posterior_corr = Sigma_mean[0,1] / np.sqrt(Sigma_mean[0,0] * Sigma_mean[1,1])

print()
print('posterior correlation between components:')
print(f'{posterior_corr:.4f}')

#----------------------------------------------------------------
# Diagnostics
#----------------------------------------------------------------
import arviz as az

# helper: lag autocorrelation (same style as your previous code)
def lag_autocorr(x, lag):
	x = np.asarray(x)
	n = len(x)
	x_mean = np.mean(x)
	c0 = np.sum((x - x_mean) ** 2) / n
	c_lag = np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / (n - lag)
	return c_lag / c0

# helper: ess using arviz (expects shape (chains, draws))
def ess(x):
	x = np.asarray(x).reshape((1, -1))
	return float(az.ess(x))

acf_lag = 10

print()
print('diagnostics for theta_1:')
print(f' - ess = {ess(theta_samples[:,0]):.0f}')
print(f' - acf_{acf_lag} = {lag_autocorr(theta_samples[:,0], acf_lag):.4f}')

print()
print('diagnostics for theta_2:')
print(f' - ess = {ess(theta_samples[:,1]):.0f}')
print(f' - acf_{acf_lag} = {lag_autocorr(theta_samples[:,1], acf_lag):.4f}')

print()
print('diagnostics for Sigma_11:')
print(f' - ess = {ess(sigma_samples[:,0]):.0f}')
print(f' - acf_{acf_lag} = {lag_autocorr(sigma_samples[:,0], acf_lag):.4f}')

print()
print('diagnostics for Sigma_22:')
print(f' - ess = {ess(sigma_samples[:,3]):.0f}')
print(f' - acf_{acf_lag} = {lag_autocorr(sigma_samples[:,3], acf_lag):.4f}')

#----------------------------------------------------------------
# Visualization
#----------------------------------------------------------------

figsize = (12,8)

# trace plots for theta components
plt.figure(figsize=figsize)
plt.plot(theta_samples[:, 0])
plt.title('trace plot: theta_1')
plt.xlabel('iteration')
plt.ylabel('value')

plt.figure(figsize=figsize)
plt.plot(theta_samples[:, 1])
plt.title('trace plot: theta_2')
plt.xlabel('iteration')
plt.ylabel('value')

# trace plot for Sigma entries
plt.figure(figsize=figsize)
plt.plot(sigma_samples[:, 0])
plt.title('trace plot: Sigma_11')

plt.figure(figsize=figsize)
plt.plot(sigma_samples[:, 3])
plt.title('trace plot: Sigma_22')

# scatter plot of (theta_1, theta_2)
plt.figure(figsize=figsize)
plt.scatter(theta_samples[:, 0], theta_samples[:, 1], s=5)

# add line theta_1 = theta_2
min_val = min(theta_samples.min(), theta_samples.min())
max_val = max(theta_samples.max(), theta_samples.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.title('posterior samples of (theta_1, theta_2)')
plt.xlabel('theta_1')
plt.ylabel('theta_2')

# # heatmap via 2d histogram
# plt.figure(figsize=figsize)
# h, xedges, yedges = np.histogram2d(theta_samples[:, 0], theta_samples[:, 1], bins=50)
# plt.imshow(h.T, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
# plt.title('posterior density heatmap of (theta_1, theta_2)')
# plt.xlabel('theta_1')
# plt.ylabel('theta_2')

plt.tight_layout()
plt.show()

#----------------------------------------------------------------
# interpretation notes (for students)
#----------------------------------------------------------------
# trace plots should look like noisy stationary sequences if the chain mixes well
# the scatter plot shows posterior dependence between theta_1 and theta_2
# the heatmap approximates the joint posterior density
# improvement distribution tells us whether instruction helps on average
