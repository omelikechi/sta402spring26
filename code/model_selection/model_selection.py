#!/usr/bin/env python3

import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import invgamma
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ============================================================
# Helpers
# ============================================================

def standardize_columns(X):
	X = np.asarray(X, dtype=float)
	mu = X.mean(axis=0)
	sd = X.std(axis=0, ddof=0)
	sd[sd == 0] = 1.0
	return (X - mu) / sd, mu, sd


def standardize_vector(y):
	y = np.asarray(y, dtype=float)
	mu = y.mean()
	sd = y.std(ddof=0)
	if sd == 0:
		sd = 1.0
	return (y - mu) / sd, mu, sd


def ssr(y, X, beta):
	resid = y - X @ beta
	return float(resid @ resid)


def ols_fit(X, y):
	beta = np.linalg.solve(X.T @ X, X.T @ y)
	resid = y - X @ beta
	n, p = X.shape
	sigma2_hat = float(resid @ resid) / (n - p)
	cov_beta = sigma2_hat * np.linalg.inv(X.T @ X)
	se_beta = np.sqrt(np.diag(cov_beta))
	return {
		'beta': beta,
		'sigma2_hat': sigma2_hat,
		'se_beta': se_beta,
		'resid': resid,
	}


def mse(y_true, y_pred):
	err = np.asarray(y_true) - np.asarray(y_pred)
	return float(np.mean(err ** 2))


# ============================================================
# Part 1: Oxygen uptake example from the chapter
# ============================================================

def oxygen_data():
	age = np.array([23, 22, 22, 25, 27, 20, 31, 23, 27, 28, 22, 24], dtype=float)
	y = np.array([-0.87, -10.74, -3.27, -1.97, 7.50, -7.25, 17.05, 4.96, 10.40, 11.05, 0.26, 2.51], dtype=float)

	# First 6 = running, last 6 = aerobics
	aerobic = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)

	X = np.column_stack([
		np.ones(len(y)),
		aerobic,
		age,
		aerobic * age,
	])
	return X, y, age, aerobic


def g_prior_posterior_samples(X, y, g=None, nu0=1.0, s20=None, S=5000, seed=123):
	"""
	Draw independent posterior samples under the invariant g-prior:
		beta | sigma2 ~ N(0, g * sigma2 * (X'X)^(-1))
		1/sigma2 ~ Gamma(nu0/2, nu0*s20/2)

	This matches the chapter's simple g-prior setup. 
	"""
	rng = np.random.default_rng(seed)
	n, p = X.shape

	XtX_inv = np.linalg.inv(X.T @ X)
	beta_ols = XtX_inv @ X.T @ y
	resid = y - X @ beta_ols
	sigma2_ols = float(resid @ resid) / (n - p)

	if g is None:
		g = n
	if s20 is None:
		s20 = sigma2_ols

	H_g = (g / (g + 1.0)) * X @ XtX_inv @ X.T
	SSR_g = float(y.T @ (np.eye(n) - H_g) @ y)

	alpha = 0.5 * (nu0 + n)
	scale = 0.5 * (nu0 * s20 + SSR_g)

	sigma2 = invgamma.rvs(a=alpha, scale=scale, size=S, random_state=rng)
	mean_beta = (g / (g + 1.0)) * beta_ols
	cov_base = (g / (g + 1.0)) * XtX_inv

	beta_samples = np.zeros((S, p))
	for s in range(S):
		beta_samples[s] = rng.multivariate_normal(mean_beta, sigma2[s] * cov_base)

	return {
		'beta_samples': beta_samples,
		'sigma2_samples': sigma2,
		'beta_ols': beta_ols,
		'sigma2_ols': sigma2_ols,
	}


def run_oxygen_demo():
	X, y, age, aerobic = oxygen_data()

	print('\n' + '=' * 60)
	print('OXYGEN EXAMPLE')
	print('=' * 60)

	ols = ols_fit(X, y)
	print('OLS beta:')
	print(np.round(ols['beta'], 2))
	print('OLS sigma^2 hat:')
	print(round(ols['sigma2_hat'], 2))
	print('OLS standard errors:')
	print(np.round(ols['se_beta'], 2))

	post = g_prior_posterior_samples(X, y, g=len(y), nu0=1.0, s20=ols['sigma2_hat'], S=4000, seed=123)
	beta_post_mean = post['beta_samples'].mean(axis=0)
	beta_post_sd = post['beta_samples'].std(axis=0, ddof=1)

	print('\nPosterior mean of beta (g-prior):')
	print(np.round(beta_post_mean, 2))
	print('Posterior SD of beta:')
	print(np.round(beta_post_sd, 2))

	ages_grid = np.arange(20, 31)
	delta_samples = post['beta_samples'][:, 1][:, None] + post['beta_samples'][:, 3][:, None] * ages_grid[None, :]
	delta_low = np.quantile(delta_samples, 0.025, axis=0)
	delta_mid = np.quantile(delta_samples, 0.5, axis=0)
	delta_high = np.quantile(delta_samples, 0.975, axis=0)

	plt.figure(figsize=(7, 4))
	plt.fill_between(ages_grid, delta_low, delta_high, alpha=0.25)
	plt.plot(ages_grid, delta_mid)
	plt.axhline(0.0, linestyle='--')
	plt.xlabel('Age')
	plt.ylabel('Aerobics effect: beta_2 + beta_4 * age')
	plt.title('Oxygen example: posterior interval for program effect')
	plt.tight_layout()
	plt.show()


# ============================================================
# Part 2: Diabetes-style model selection example
# ============================================================

def build_diabetes_design():
	"""
	Build the 64 regressors described in the chapter:
	- 10 main effects
	- 45 pairwise interactions
	- 9 quadratic terms (omit x_2^2 because x_2 = sex is binary in the chapter)
	
	sklearn's diabetes data are not literally the same preprocessed matrix as in the book,
	but this is close and pedagogically clean.
	"""
	data = load_diabetes()
	X0 = data.data.copy()
	y0 = data.target.copy()

	p0 = X0.shape[1]
	names0 = list(data.feature_names)

	cols = []
	names = []

	# Main effects
	for j in range(p0):
		cols.append(X0[:, j])
		names.append(names0[j])

	# Pairwise interactions
	for j, k in itertools.combinations(range(p0), 2):
		cols.append(X0[:, j] * X0[:, k])
		names.append(f'{names0[j]}:{names0[k]}')

	# Quadratics, omitting x_2^2
	# In sklearn feature order, sex is the second variable.
	sex_index = 1
	for j in range(p0):
		if j == sex_index:
			continue
		cols.append(X0[:, j] ** 2)
		names.append(f'{names0[j]}^2')

	X = np.column_stack(cols)
	X, _, _ = standardize_columns(X)
	y, _, _ = standardize_vector(y0)

	return X, y, names


def log_marginal_y_given_X_gprior(y, X, g=None, nu0=1.0, s20=1.0):
	"""
	Log p(y | X, model) under the invariant g-prior after integrating out beta and sigma^2.

	For teaching:
	- We omit constants that cancel across models where convenient.
	- This is enough for Gibbs updates because only log-ratios matter.
	"""
	n = len(y)

	if X.shape[1] == 0:
		SSR_g = float(y @ y)
		p = 0
	else:
		p = X.shape[1]
		if g is None:
			g = n

		XtX_inv = np.linalg.inv(X.T @ X)
		H_g = (g / (g + 1.0)) * X @ XtX_inv @ X.T
		SSR_g = float(y.T @ (np.eye(n) - H_g) @ y)

	alpha = 0.5 * (nu0 + n)
	# Up to additive constants:
	# log p(y | X) = -(p/2) log(1+g) - alpha * log(nu0*s20 + SSR_g) + const
	return -0.5 * p * np.log(1.0 + (n if g is None else g)) - alpha * np.log(nu0 * s20 + SSR_g)


def posterior_beta_given_model(y, X, g=None, nu0=1.0, s20=1.0, rng=None):
	"""
	Draw one posterior sample of beta for a fixed model under the g-prior.
	"""
	if rng is None:
		rng = np.random.default_rng()

	n = len(y)
	p = X.shape[1]

	if p == 0:
		return np.zeros(0), invgamma.rvs(a=0.5 * (nu0 + n), scale=0.5 * (nu0 * s20 + y @ y), random_state=rng)

	if g is None:
		g = n

	XtX_inv = np.linalg.inv(X.T @ X)
	beta_ols = XtX_inv @ X.T @ y
	H_g = (g / (g + 1.0)) * X @ XtX_inv @ X.T
	SSR_g = float(y.T @ (np.eye(n) - H_g) @ y)

	sigma2 = invgamma.rvs(
		a=0.5 * (nu0 + n),
		scale=0.5 * (nu0 * s20 + SSR_g),
		random_state=rng,
	)

	mean_beta = (g / (g + 1.0)) * beta_ols
	cov_beta = (g / (g + 1.0)) * sigma2 * XtX_inv
	beta = rng.multivariate_normal(mean_beta, cov_beta)
	return beta, sigma2


def gibbs_model_selection(y, X, S=10000, g=None, nu0=1.0, s20=1.0, prior_inclusion=0.5, seed=123):
	"""
	Simple Gibbs sampler on z in {0,1}^p.

	For each coordinate j:
	- compare current model vs toggled model
	- use Bernoulli full conditional based on posterior odds
	"""
	rng = np.random.default_rng(seed)
	n, p = X.shape

	if g is None:
		g = n

	z = np.zeros(p, dtype=int)
	beta_full_samples = np.zeros((S, p))
	z_samples = np.zeros((S, p), dtype=int)

	def current_log_post(z_vec):
		idx = np.flatnonzero(z_vec)
		X_sub = X[:, idx]
		log_like = log_marginal_y_given_X_gprior(y, X_sub, g=g, nu0=nu0, s20=s20)

		k = idx.size
		log_prior = k * np.log(prior_inclusion) + (p - k) * np.log(1.0 - prior_inclusion)
		return log_like + log_prior

	log_post_current = current_log_post(z)

	for s in range(S):
		for j in rng.permutation(p):
			z_prop = z.copy()
			z_prop[j] = 1 - z_prop[j]

			log_post_prop = current_log_post(z_prop)
			logit = log_post_prop - log_post_current
			prob_one = 1.0 / (1.0 + np.exp(-logit))

			z[j] = rng.binomial(1, prob_one)
			if z[j] == z_prop[j]:
				log_post_current = log_post_prop

		idx = np.flatnonzero(z)
		beta_s = np.zeros(p)

		if idx.size > 0:
			beta_sub, _ = posterior_beta_given_model(y, X[:, idx], g=g, nu0=nu0, s20=s20, rng=rng)
			beta_s[idx] = beta_sub

		beta_full_samples[s] = beta_s
		z_samples[s] = z

	return {
		'z_samples': z_samples,
		'beta_samples': beta_full_samples,
		'pip': z_samples.mean(axis=0),
		'beta_bma': beta_full_samples.mean(axis=0),
	}


def run_diabetes_demo(seed=123):
	X, y, names = build_diabetes_design()

	# Same split sizes as the chapter: 342 train, 100 test
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=100,
		random_state=seed,
	)

	print('\n' + '=' * 60)
	print('DIABETES MODEL SELECTION EXAMPLE')
	print('=' * 60)
	print(f'Training sample size: {X_train.shape[0]}')
	print(f'Test sample size: {X_test.shape[0]}')
	print(f'Number of regressors: {X_train.shape[1]}')

	# Full OLS benchmark
	ols = LinearRegression(fit_intercept=False)
	ols.fit(X_train, y_train)
	yhat_full = ols.predict(X_test)
	mse_full = mse(y_test, yhat_full)
	print(f'\nFull OLS test MSE: {mse_full:.3f}')

	# Bayesian model selection
	out = gibbs_model_selection(
		y=y_train,
		X=X_train,
		S=6000,
		g=X_train.shape[0],
		nu0=1.0,
		s20=1.0,
		prior_inclusion=0.5,
		seed=seed,
	)

	pip = out['pip']
	beta_bma = out['beta_bma']
	yhat_bma = X_test @ beta_bma
	mse_bma = mse(y_test, yhat_bma)

	selected = np.where(pip > 0.5)[0]
	print(f'Bayesian model-averaged test MSE: {mse_bma:.3f}')
	print(f'Number of regressors with PIP > 0.5: {len(selected)}')

	if len(selected) > 0:
		order = selected[np.argsort(-pip[selected])]
		print('\nTop selected regressors:')
		for idx in order[:15]:
			print(f'  {names[idx]:20s}  PIP = {pip[idx]:.3f}')

	# Permuted response sanity check
	rng = np.random.default_rng(seed + 1)
	y_perm = rng.permutation(y_train)
	out_perm = gibbs_model_selection(
		y=y_perm,
		X=X_train,
		S=4000,
		g=X_train.shape[0],
		nu0=1.0,
		s20=1.0,
		prior_inclusion=0.5,
		seed=seed + 1,
	)
	pip_perm = out_perm['pip']

	print('\nPermuted-response check:')
	print(f'  max PIP = {pip_perm.max():.3f}')
	print(f'  number with PIP > 0.5 = {np.sum(pip_perm > 0.5)}')

	# Plots
	plt.figure(figsize=(7, 4))
	plt.plot(np.arange(1, len(pip) + 1), pip, 'o')
	plt.axhline(0.5, linestyle='--')
	plt.xlabel('Regressor index')
	plt.ylabel('Posterior inclusion probability')
	plt.title('Diabetes example: posterior inclusion probabilities')
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(5, 5))
	plt.scatter(y_test, yhat_bma)
	lims = [min(y_test.min(), yhat_bma.min()), max(y_test.max(), yhat_bma.max())]
	plt.plot(lims, lims, linestyle='--')
	plt.xlabel('y_test')
	plt.ylabel('Model-averaged prediction')
	plt.title('Diabetes example: test predictions')
	plt.tight_layout()
	plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
	run_oxygen_demo()
	run_diabetes_demo(seed=123)