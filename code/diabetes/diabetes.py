# Linear regression for diabetes data
"""
This script uses the classic diabetes dataset available from
scikit‑learn (originally from Efron et al., 2004). The dataset
contains measurements on 442 diabetes patients.

Outcome variable:
	target — quantitative measure of disease progression one year
			after baseline.

The outcome is a continuous diabetes progression score derived
from clinical measurements including blood glucose and related
biomarkers. Larger values correspond to worse diabetes progression.

Possible predictors (all standardized to mean 0 and variance 1):
	age  — age of patient
	sex  — sex of patient
	bmi  — body mass index
	bp   — average blood pressure
	s1   — total serum cholesterol
	s2   — low‑density lipoproteins
	s3   — high‑density lipoproteins
	s4   — total cholesterol / HDL ratio
	s5   — log serum triglycerides
	s6   — blood sugar level

Notes:
	• Predictors are standardized (mean 0, variance 1)
	• 10 predictors total
	• n = 442 samples
	• Dataset introduced in Efron et al. (2004) for regression benchmarking
"""

#----------------------------------------------------------------
# Bayesian / Classical Linear Regression
#   1. Classical OLS
#   2. Bayesian linear regression with independent priors
#   3. Bayesian linear regression with Zellner's g‑prior (g = n)
#----------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.linalg import solve
from sklearn.datasets import load_diabetes

np.random.seed(302)

#----------------------------------------------------------------
# User options
#----------------------------------------------------------------

MODEL_SPECS = {
	'all': ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'],
	'bmi': ['bmi'],
	'age': ['age'],
	'bp': ['bp'],
	'bmi_age': ['bmi', 'age'],
	'bmi_age_bp': ['bmi', 'age', 'bp']
}

model_to_fit = 'bmi'

# Optionally subsample
n_samples = None

n_gibbs = 6000
burn = 1000

plot_posterior_predictive_interval = True

print('\nModel:', model_to_fit)

#----------------------------------------------------------------
# Helpers
#----------------------------------------------------------------
def summarize_vector(samples, name):
	q = np.quantile(samples, [0.025, 0.5, 0.975])
	return {
		'parameter': name,
		'mean': samples.mean(),
		'sd': samples.std(ddof=1),
		'q2.5': q[0],
		'median': q[1],
		'q97.5': q[2]
	}

def print_summary_table(summary_df, title):
	print(f'\n{title}')
	print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

def posterior_mean_interval(beta_samples, X):
	draws = np.dot(beta_samples, X.T)
	return {
		'mean': draws.mean(axis=0),
		'lower': np.quantile(draws, 0.025, axis=0),
		'upper': np.quantile(draws, 0.975, axis=0)
	}

def posterior_predictive_interval(beta_samples, sigma2_samples, X):
	mean_draws = np.dot(beta_samples, X.T)
	noise = np.random.normal(
		loc=0,
		scale=np.sqrt(sigma2_samples)[:, None],
		size=mean_draws.shape
	)
	predictive_draws = mean_draws + noise
	return {
		'mean': predictive_draws.mean(axis=0),
		'lower': np.quantile(predictive_draws, 0.025, axis=0),
		'upper': np.quantile(predictive_draws, 0.975, axis=0)
	}

#----------------------------------------------------------------
# Load data
#----------------------------------------------------------------
data = load_diabetes(as_frame=True)

df = data.frame.copy()

outcome_name = 'target'

predictor_cols = MODEL_SPECS[model_to_fit]

analysis_cols = predictor_cols + [outcome_name]
analysis_df = df[analysis_cols].dropna().copy()

if n_samples is not None and n_samples < len(analysis_df):
	analysis_df = analysis_df.sample(n=n_samples, random_state=302)
	print(f'Using random subset of size {n_samples}')

#----------------------------------------------------------------
# Build design matrix
#----------------------------------------------------------------
X_df = analysis_df[predictor_cols].astype(float)

# Rescale predictors to mean 0 and unit variance
predictor_means = X_df.mean()
predictor_stds = X_df.std()
X_df = (X_df - predictor_means) / predictor_stds

X_df.insert(0, 'intercept', 1.0)

X = X_df.to_numpy()
y = analysis_df[outcome_name].to_numpy()

coef_names = list(X_df.columns)

print('\nSample size:', len(y))
print('Design columns:', coef_names)
print(f'Outcome mean: {y.mean():.2f}')
print(f'Outcome variance: {y.var(ddof=1):.2f}')

#----------------------------------------------------------------
# OLS
#----------------------------------------------------------------
def fit_ols(X, y):
	XtX = np.dot(X.T, X)
	XtY = np.dot(X.T, y)
	XtX_inv = np.linalg.inv(XtX)

	beta_hat = solve(XtX, XtY, assume_a='sym')

	n, p = X.shape
	resid = y - np.dot(X, beta_hat)
	sse = np.dot(resid, resid)

	sigma2_hat = sse / (n - p)
	cov_beta_hat = sigma2_hat * XtX_inv

	return {
		'beta_hat': beta_hat,
		'sigma2_hat': sigma2_hat,
		'cov_beta_hat': cov_beta_hat
	}

ols = fit_ols(X, y)

ols_summary = pd.DataFrame([
	{
		'parameter': name,
		'estimate': ols['beta_hat'][j],
		'se': np.sqrt(ols['cov_beta_hat'][j, j])
	}
	for j, name in enumerate(coef_names)
])

print_summary_table(ols_summary, 'OLS summary')
print(f'OLS sigma estimate: {np.sqrt(ols["sigma2_hat"]):.2f}')

#----------------------------------------------------------------
# Bayesian independent priors
#----------------------------------------------------------------
"""
Bayesian linear regression with independent priors
--------------------------------
Model:
	y = X beta + epsilon
	epsilon ~ N(0, sigma^2 I)
Prior:
	sigma^2 ~ Inverse-Gamma(a0, b0)
	beta | sigma^2 ~ N(beta0, V0)
where
	beta0 = (ȳ, 0, ..., 0)
	V0 = diag(1e6, 1e6, ..., 1e6)
This prior:
• Assumes coefficients are independent a priori
• Shrinks slopes toward zero
• Places minimal information on coefficient magnitude
• Does not depend on the design matrix
Because the variance is very large (1e6), the prior behaves similarly
to a noninformative prior.
with
	a0 = 0.01
	b0 = 0.01
This leads to conditional posteriors:
	beta | sigma^2, y ~ N(m_n, V_n)
	sigma^2 | beta, y ~ Inverse-Gamma(a_n, b_n)
These are sampled using Gibbs sampling.
This specification corresponds to standard Bayesian linear regression
with diffuse independent priors on coefficients.
"""
a0 = 0.01
b0 = 0.01

prior_var_intercept = 1e6
prior_var_slope = 1e6

def beta_full_conditional(X, y, sigma2, beta0, V0):
	V0_inv = np.linalg.inv(V0)
	XtX = np.dot(X.T, X)
	XtY = np.dot(X.T, y)
	Vn_inv = V0_inv + XtX / sigma2
	Vn = np.linalg.inv(Vn_inv)

	mn = np.dot(Vn, np.dot(V0_inv, beta0) + XtY / sigma2)
	return mn, Vn


def sigma2_full_conditional(X, y, beta, a0, b0):
	resid = y - np.dot(X, beta)
	an = a0 + len(y) / 2
	bn = b0 + 0.5 * np.dot(resid, resid)
	return an, bn


def gibbs_independent_priors(X, y, beta0, V0, a0, b0, n_gibbs, burn):
	n, p = X.shape

	beta_samples = np.zeros((n_gibbs, p))
	sigma2_samples = np.zeros(n_gibbs)

	ols = fit_ols(X, y)

	beta_curr = ols['beta_hat']
	sigma2_curr = ols['sigma2_hat']

	for s in range(n_gibbs):

		mn, Vn = beta_full_conditional(X, y, sigma2_curr, beta0, V0)

		beta_curr = np.random.multivariate_normal(mn, Vn)

		an, bn = sigma2_full_conditional(
			X, y, beta_curr, a0, b0
		)

		sigma2_curr = invgamma.rvs(a=an, scale=bn)

		beta_samples[s] = beta_curr
		sigma2_samples[s] = sigma2_curr

	return beta_samples[burn:], sigma2_samples[burn:]

p = X.shape[1]

beta0 = np.zeros(p)
beta0[0] = y.mean()
V0 = np.diag([prior_var_intercept] + [prior_var_slope] * (p - 1))

beta_indep, sigma2_indep = gibbs_independent_priors(
	X, y, beta0, V0, a0, b0, n_gibbs, burn
)

indep_rows = []

for j, name in enumerate(coef_names):
	indep_rows.append(summarize_vector(beta_indep[:, j], name))

indep_rows.append(summarize_vector(np.sqrt(sigma2_indep), 'sigma'))

indep_summary = pd.DataFrame(indep_rows)

print_summary_table(
	indep_summary,
	'Bayesian linear regression: independent priors'
)

#----------------------------------------------------------------
# Zellner g‑prior
"""
Bayesian linear regression with Zellner's g‑prior
--------------------------------
Model:
	y = X beta + epsilon
	epsilon ~ N(0, sigma^2 I)
Prior:
	sigma^2 ~ Inverse‑Gamma(n/2, b_n)
	beta | sigma^2 ~ N(0, g sigma^2 (X^T X)^(-1))
where
	g = n  (sample size)

This prior:
• Scales with the design matrix
• Is invariant to linear transformations of predictors
• Shrinks coefficients toward zero
• Uses data‑dependent covariance structure

This leads to a closed‑form posterior:
	beta | sigma^2, y ~ N(beta_n, sigma^2 V_n)
with shrinkage factor
	beta_n = (g / (g + 1)) beta_hat
Thus the posterior mean is a shrunk version of the OLS estimator.
Using g = n corresponds to the "unit information prior"
and is a common default choice.
"""
#----------------------------------------------------------------

def fit_g_prior(X, y, g=None):

	n, p = X.shape

	if g is None:
		g = n

	XtX = np.dot(X.T, X)
	XtY = np.dot(X.T, y)

	XtX_inv = np.linalg.inv(XtX)

	beta_hat = solve(XtX, XtY, assume_a='sym')

	beta_n = (g / (g + 1)) * beta_hat

	resid = y - np.dot(X, beta_hat)
	sse = np.dot(resid, resid)

	quad = np.dot(beta_hat, np.dot(XtX, beta_hat))

	a_n = n / 2
	b_n = 0.5 * (sse + quad / (g + 1))

	V_n = (g / (g + 1)) * XtX_inv

	return {
		'beta_n': beta_n,
		'V_n': V_n,
		'a_n': a_n,
		'b_n': b_n
	}


def sample_g_prior_posterior(beta_n, V_n, a_n, b_n, S):

	sigma2 = invgamma.rvs(a=a_n, scale=b_n, size=S)
	p = len(beta_n)

	beta = np.zeros((S, p))

	for s in range(S):
		beta[s] = np.random.multivariate_normal(
			beta_n,
			sigma2[s] * V_n
		)

	return beta, sigma2


g_prior = fit_g_prior(X, y, g=len(y))

beta_g, sigma2_g = sample_g_prior_posterior(
	g_prior['beta_n'],
	g_prior['V_n'],
	g_prior['a_n'],
	g_prior['b_n'],
	len(sigma2_indep)
)

g_rows = []

for j, name in enumerate(coef_names):
	g_rows.append(summarize_vector(beta_g[:, j], name))

g_rows.append(summarize_vector(np.sqrt(sigma2_g), 'sigma'))

g_summary = pd.DataFrame(g_rows)

print_summary_table(
	g_summary,
	"Bayesian linear regression: Zellner's g‑prior (g = n)"
)

#----------------------------------------------------------------
# Visualization
#----------------------------------------------------------------

import matplotlib.pyplot as plt

if X.shape[1] == 2:

	x = X[:, 1]

	order = np.argsort(x)
	x_sorted = x[order]

	y_ols = np.dot(X, ols['beta_hat'])[order]

	indep_mean_interval = posterior_mean_interval(beta_indep, X)
	g_mean_interval = posterior_mean_interval(beta_g, X)

	if plot_posterior_predictive_interval:
		indep_predictive_interval = posterior_predictive_interval(beta_indep, sigma2_indep, X)
		g_predictive_interval = posterior_predictive_interval(beta_g, sigma2_g, X)

	fig, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)
	ax1, ax2 = axes

	#------------------------------------------------------------
	# Mean function
	#------------------------------------------------------------
	ax1.scatter(x, y, alpha=0.3, label='Data')
	ax1.plot(x_sorted, y_ols, linewidth=3, label='OLS')

	ax1.plot(
		x_sorted,
		indep_mean_interval['mean'][order],
		linewidth=3,
		label='Bayes (independent)'
	)

	ax1.fill_between(
		x_sorted,
		indep_mean_interval['lower'][order],
		indep_mean_interval['upper'][order],
		alpha=0.2
	)

	ax1.plot(
		x_sorted,
		g_mean_interval['mean'][order],
		linewidth=3,
		label='Bayes (g‑prior)'
	)

	ax1.fill_between(
		x_sorted,
		g_mean_interval['lower'][order],
		g_mean_interval['upper'][order],
		alpha=0.2
	)

	#------------------------------------------------------------
	# Posterior predictive
	#------------------------------------------------------------
	ax2.scatter(x, y, alpha=0.3, label='Data')
	ax2.plot(x_sorted, y_ols, linewidth=3, label='OLS')

	# mean regression lines
	ax2.plot(
		x_sorted,
		indep_mean_interval['mean'][order],
		linewidth=3,
		label='Bayes mean (independent)'
	)

	ax2.plot(
		x_sorted,
		g_mean_interval['mean'][order],
		linewidth=3,
		label='Bayes mean (g‑prior)'
	)

	if plot_posterior_predictive_interval:

		ax2.fill_between(
			x_sorted,
			indep_predictive_interval['lower'][order],
			indep_predictive_interval['upper'][order],
			alpha=0.2,
			label='Bayes (independent)'
		)

		ax2.fill_between(
			x_sorted,
			g_predictive_interval['lower'][order],
			g_predictive_interval['upper'][order],
			alpha=0.2,
			label='Bayes (g‑prior)'
		)

	ax1.set_xlabel(predictor_cols[0])
	ax2.set_xlabel(predictor_cols[0])
	ax1.set_ylabel(outcome_name)

	ax1.set_title('Mean function (credible intervals)')
	ax2.set_title('Posterior predictive intervals')

	ax1.legend()
	ax2.legend()

	plt.tight_layout()
	plt.show()

else:

	coef_indep = beta_indep[:, 1:]
	coef_g = beta_g[:, 1:]

	fig, axes = plt.subplots(2, 1, figsize=(14,9), sharex=True)

	axes[0].axhline(0, linestyle='--', color='gray', alpha=0.75)
	axes[0].boxplot(coef_indep, tick_labels=predictor_cols, showfliers=False)
	axes[0].set_title('Bayesian independent priors — coefficient distributions')
	axes[0].set_ylabel('Effect on diabetes')

	axes[1].axhline(0, linestyle='--', color='gray', alpha=0.75)
	axes[1].boxplot(coef_g, tick_labels=predictor_cols, showfliers=False)
	axes[1].set_title("Zellner's g-prior — coefficient distributions")
	axes[1].set_ylabel('Effect on diabetes')

	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()


