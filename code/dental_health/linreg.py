
import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.linalg import solve

#----------------------------------------------------------------
# Linear regression for NHANES dental data
#   1. Classical OLS
#   2. Bayesian linear regression with independent priors
#   3. Bayesian linear regression with Zellner's g-prior (g = n)
#----------------------------------------------------------------

np.random.seed(302)

#----------------------------------------------------------------
# User options
#----------------------------------------------------------------

show_nonlinear = True
age_min = None
age_max = None
n_samples = None

dentition_path = 'OHXDEN_H.XPT'
demographics_path = 'DEMO_H.XPT'

codes_to_count = ['D', 'M', 'R']
outcome_name = 'count_' + ''.join(codes_to_count)

MODEL_SPECS = {
	'age_sex': ['RIDAGEYR', 'RIAGENDR'],
	'age': ['RIDAGEYR'],
	'income': ['INDFMPIR'],
	'sex': ['RIAGENDR'],
	'age_sex_income': ['RIDAGEYR', 'RIAGENDR', 'INDFMPIR'],
	'age_sex_education': ['RIDAGEYR', 'RIAGENDR', 'DMDEDUC2']
}

model_to_fit = 'age'
predictor_cols = MODEL_SPECS[model_to_fit]

# Independent-prior model:
#   beta ~ N(beta0, V0)
#   sigma^2 ~ IG(a0, b0)
#
# Since beta | sigma^2 does not scale with sigma^2 here,
# we use a Gibbs sampler based on the full conditionals.
a0 = 2.0
b0 = 25.0
prior_var_intercept = 100.0
prior_var_slope = 25.0

n_gibbs = 6000
burn = 1000

print('\nModel:', model_to_fit)
print('Predictors:', predictor_cols)

#----------------------------------------------------------------
# Helpers
#----------------------------------------------------------------
def make_outcome(df, codes):
	ctc_vars = [c for c in df.columns if c.endswith('CTC')]

	for c in ctc_vars:
		df[c] = df[c].map(lambda x: x.decode() if isinstance(x, bytes) else x)

	df[outcome_name] = df[ctc_vars].isin(codes).sum(axis=1)
	return df


def prepare_data():
	dental = pd.read_sas(dentition_path)
	dental = make_outcome(dental, codes_to_count)

	demo = pd.read_sas(demographics_path)
	df = dental.merge(demo, on='SEQN', how='inner')

	if age_min is not None:
		df = df[df['RIDAGEYR'] >= age_min]
	if age_max is not None:
		df = df[df['RIDAGEYR'] <= age_max]

	return df.reset_index(drop=True)


def build_design_matrix(df, predictors):
	X_df = df[predictors].copy()

	if 'RIAGENDR' in X_df.columns:
		X_df['male'] = (X_df['RIAGENDR'] == 1).astype(float)
		X_df = X_df.drop(columns=['RIAGENDR'])

	X_df = X_df.astype(float)
	X_df.insert(0, 'intercept', 1.0)

	y = df[outcome_name].astype(float).to_numpy()
	X = X_df.to_numpy()

	return X_df, X, y


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
		'cov_beta_hat': cov_beta_hat,
		'resid': resid,
		'sse': sse,
		'XtX_inv': XtX_inv
	}


#----------------------------------------------------------------
# Bayesian linear regression with independent priors
#----------------------------------------------------------------
# beta | sigma^2, y, X is multivariate normal
# sigma^2 | beta, y, X is inverse-gamma
#----------------------------------------------------------------

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
	beta_curr = ols['beta_hat'].copy()
	sigma2_curr = ols['sigma2_hat']

	for s in range(n_gibbs):
		mn, Vn = beta_full_conditional(X, y, sigma2_curr, beta0, V0)
		beta_curr = np.random.multivariate_normal(mn, Vn)

		an, bn = sigma2_full_conditional(X, y, beta_curr, a0, b0)
		sigma2_curr = invgamma.rvs(a=an, scale=bn)

		beta_samples[s] = beta_curr
		sigma2_samples[s] = sigma2_curr

	return beta_samples[burn:], sigma2_samples[burn:]


#----------------------------------------------------------------
# Zellner's g-prior with g = n
#----------------------------------------------------------------
# Prior:
#   beta | sigma^2 ~ N(0, g sigma^2 (X'X)^(-1))
#   p(sigma^2) proportional to 1 / sigma^2
# Posterior is available in closed form.
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

	resid_hat = y - np.dot(X, beta_hat)
	sse = np.dot(resid_hat, resid_hat)
	quad = np.dot(beta_hat, np.dot(XtX, beta_hat))

	a_n = n / 2
	b_n = 0.5 * (sse + quad / (g + 1))
	V_n = (g / (g + 1)) * XtX_inv

	return {
		'g': g,
		'beta_n': beta_n,
		'V_n': V_n,
		'a_n': a_n,
		'b_n': b_n
	}


def sample_g_prior_posterior(beta_n, V_n, a_n, b_n, S):
	p = len(beta_n)
	sigma2 = invgamma.rvs(a=a_n, scale=b_n, size=S)
	beta = np.zeros((S, p))

	for s in range(S):
		beta[s] = np.random.multivariate_normal(beta_n, sigma2[s] * V_n)

	return beta, sigma2


#----------------------------------------------------------------
# Load data and inspect predictors
#----------------------------------------------------------------

df = prepare_data()

#----------------------------------------------------------------
# Build analysis dataset
#----------------------------------------------------------------

analysis_cols = predictor_cols + [outcome_name]
analysis_df = df[analysis_cols].dropna().copy()

# Optional subsampling
if n_samples is not None and n_samples < len(analysis_df):
	analysis_df = analysis_df.sample(n=n_samples, random_state=302).reset_index(drop=True)
	print(f'Using random subset of size {n_samples}')

X_df, X, y = build_design_matrix(analysis_df, predictor_cols)
coef_names = list(X_df.columns)

print('\nSample size:', len(y))
print('Design columns:', coef_names)
print(f'Outcome mean: {y.mean():.2f}')
print(f'Outcome variance: {y.var(ddof=1):.2f}')

#----------------------------------------------------------------
# Fit OLS
#----------------------------------------------------------------

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
print(f'\nOLS sigma^2 estimate: {ols["sigma2_hat"]:.2f}')

#----------------------------------------------------------------
# Fit Bayesian model with independent priors
#----------------------------------------------------------------

p = X.shape[1]
beta0 = np.zeros(p)
V0 = np.diag([prior_var_intercept] + [prior_var_slope] * (p - 1))

beta_indep, sigma2_indep = gibbs_independent_priors(
	X, y, beta0, V0, a0, b0, n_gibbs, burn
)

indep_rows = []
for j, name in enumerate(coef_names):
	indep_rows.append(summarize_vector(beta_indep[:, j], name))
indep_rows.append(summarize_vector(sigma2_indep, 'sigma2'))

indep_summary = pd.DataFrame(indep_rows)
print_summary_table(indep_summary, 'Bayesian linear regression: independent priors')

#----------------------------------------------------------------
# Fit Bayesian model with Zellner's g-prior
#----------------------------------------------------------------

g_prior = fit_g_prior(X, y, g=len(y))

beta_g, sigma2_g = sample_g_prior_posterior(
	g_prior['beta_n'], g_prior['V_n'], g_prior['a_n'], g_prior['b_n'], len(sigma2_indep)
)

g_rows = []
for j, name in enumerate(coef_names):
	g_rows.append(summarize_vector(beta_g[:, j], name))
g_rows.append(summarize_vector(sigma2_g, 'sigma2'))

g_summary = pd.DataFrame(g_rows)
print_summary_table(g_summary, "Bayesian linear regression: Zellner's g-prior (g = n)")

#----------------------------------------------------------------
# Visualization
#----------------------------------------------------------------

import matplotlib.pyplot as plt

# Only meaningful to plot fitted line when we have 1 predictor (plus intercept)
if X.shape[1] == 2:

	x = X[:, 1]
	order = np.argsort(x)
	x_sorted = x[order]

	# OLS
	y_ols = np.dot(X, ols['beta_hat'])[order]

	# Bayesian independent priors
	beta_indep_mean = beta_indep.mean(axis=0)
	y_indep = np.dot(X, beta_indep_mean)[order]

	y_indep_draws = np.dot(beta_indep, X.T)
	y_indep_lower = np.quantile(y_indep_draws, 0.025, axis=0)[order]
	y_indep_upper = np.quantile(y_indep_draws, 0.975, axis=0)[order]

	# g‑prior
	beta_g_mean = beta_g.mean(axis=0)
	y_g = np.dot(X, beta_g_mean)[order]

	y_g_draws = np.dot(beta_g, X.T)
	y_g_lower = np.quantile(y_g_draws, 0.025, axis=0)[order]
	y_g_upper = np.quantile(y_g_draws, 0.975, axis=0)[order]

	plt.figure(figsize=(10,6))

	plt.scatter(x, y, alpha=0.3, label='Data')
	plt.plot(x_sorted, y_ols, linewidth=3, label='OLS')

	plt.plot(x_sorted, y_indep, linewidth=3, label='Bayes (independent priors)')
	plt.fill_between(x_sorted, y_indep_lower, y_indep_upper, alpha=0.2)

	plt.plot(x_sorted, y_g, linewidth=3, label='Bayes (g‑prior)')
	plt.fill_between(x_sorted, y_g_lower, y_g_upper, alpha=0.2)

	plt.xlabel(predictor_cols[0])
	plt.ylabel(outcome_name)
	plt.title('Linear regression fits')
	plt.legend()
	plt.tight_layout()
	plt.show()

else:

	# Separate regressions by sex (male indicator assumed in column 2)
	if 'male' in coef_names:

		male_idx = coef_names.index('male')
		age_idx = 1

		for sex_val, title in [(0, 'Women'), (1, 'Men')]:

			mask = X[:, male_idx] == sex_val
			X_sub = X[mask]
			y_sub = y[mask]

			# Drop male column (constant within subgroup)
			X_sub = np.delete(X_sub, male_idx, axis=1)

			# Refit models within subgroup
			ols_sub = fit_ols(X_sub, y_sub)

			# adjust priors for reduced dimension
			p_sub = X_sub.shape[1]
			beta0_sub = np.zeros(p_sub)
			V0_sub = np.diag([prior_var_intercept] + [prior_var_slope]*(p_sub-1))

			beta_indep_sub, sigma2_indep_sub = gibbs_independent_priors(
				X_sub, y_sub, beta0_sub, V0_sub, a0, b0, n_gibbs, burn
			)

			g_prior_sub = fit_g_prior(X_sub, y_sub, g=len(y_sub))

			beta_g_sub, sigma2_g_sub = sample_g_prior_posterior(
				g_prior_sub['beta_n'], g_prior_sub['V_n'],
				g_prior_sub['a_n'], g_prior_sub['b_n'],
				len(sigma2_indep_sub)
			)

			x = X_sub[:, 1]
			order = np.argsort(x)
			x_sorted = x[order]

			X_grid = X_sub[order].copy()

			y_ols = np.dot(X_grid, ols_sub['beta_hat'])

			beta_indep_mean = beta_indep_sub.mean(axis=0)
			y_indep = np.dot(X_grid, beta_indep_mean)

			beta_g_mean = beta_g_sub.mean(axis=0)
			y_g = np.dot(X_grid, beta_g_mean)

			plt.figure(figsize=(10,6))

			plt.scatter(x, y_sub, alpha=0.3, label='Data')
			plt.plot(x_sorted, y_ols, linewidth=3, label='OLS')
			plt.plot(x_sorted, y_indep, linewidth=3, label='Bayes (independent)')
			plt.plot(x_sorted, y_g, linewidth=3, label='Bayes (g‑prior)')

			plt.xlabel(coef_names[age_idx])
			plt.ylabel(outcome_name)
			plt.title(f'Linear regression: {title}')
			plt.legend()
			plt.tight_layout()
			plt.show()

	else:
		print('Plotting skipped.')

#----------------------------------------------------------------
# Optional: Nonlinear methods
#----------------------------------------------------------------
if show_nonlinear:

	from sklearn.ensemble import RandomForestRegressor

	if X.shape[1] == 2:

		rf = RandomForestRegressor()

		rf.fit(x.reshape(-1,1), y)

		x_grid = np.linspace(x.min(), x.max(), 100)
		y_rf = rf.predict(x_grid.reshape(-1,1))

		plt.figure(figsize=(10,6))

		plt.scatter(x, y, alpha=0.3, label='Data')
		plt.plot(x_sorted, y_ols, linewidth=3, label='OLS')
		plt.plot(x_sorted, y_indep, linewidth=3, label='Bayes (independent)')
		plt.plot(x_sorted, y_g, linewidth=3, label='Bayes (g‑prior)')
		plt.plot(x_grid, y_rf, linewidth=3, label='Random forest')

		#------------------------------------------------------------
		# XGBoost
		#------------------------------------------------------------
		try:
			from xgboost import XGBRegressor

			xgb = XGBRegressor()

			xgb.fit(x.reshape(-1,1), y)
			y_xgb = xgb.predict(x_grid.reshape(-1,1))
			plt.plot(x_grid, y_xgb, linewidth=3, label='XGBoost')

		except:
			pass

	plt.xlabel(predictor_cols[0])
	plt.ylabel(outcome_name)
	plt.title('Model comparison: linear vs ML methods')
	plt.legend()
	plt.tight_layout()
	plt.show()


