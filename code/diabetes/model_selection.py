# Bayesian model selection
"""
Model setup
-----------
For each subset z of the 10 predictors, we fit the Gaussian regression model
    y = alpha + x_z beta_z + epsilon,     epsilon ~ N(0, sigma^2 I)
with:
    beta_z | sigma^2, x_z ~ N(0, g sigma^2 (x_z' x_z)^{-1})
    gamma = 1 / sigma^2 ~ Gamma(nu0 / 2, nu0 * sigma0^2 / 2)
using the defaults:
    g = n
    nu0 = 1
    sigma0^2 = residual variance estimate for the current model
"""

from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.special import gammaln, logsumexp
from scipy.stats import invgamma
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

np.random.seed(302)

#---------------------------------------------------------------------
# User options
#---------------------------------------------------------------------
n_samples = None
test_size = 0.25
split_random_state = 302
top_models_to_print = 15
posterior_draws = 5000
show_plots = True
top_models_to_plot = 10

use_uniform_model_prior = True
prior_inclusion_prob = 0.5

#---------------------------------------------------------------------
# Helpers
#---------------------------------------------------------------------
def ols_residual_variance(x, y):
    n, p = x.shape

    if p == 0:
        resid = y.copy()
        df = n
    else:
        xtx = x.T @ x
        xty = np.einsum('ij,i->j', x, y)
        beta_hat = solve(xtx, xty, assume_a='sym')
        resid = y - np.einsum('ij,j->i', x, beta_hat)
        df = n - p

    return np.dot(resid, resid) / df

def log_marginal_likelihood_g_prior(y, x, g=None, nu0=1.0, sigma0_sq=None):
    """
    Section 9.3.1 log p(y | x, z) under the modified g-prior
    """
    n = len(y)
    p = x.shape[1]

    if g is None:
        g = n

    y_centered = y - y.mean()

    if p == 0:
        x_centered = np.empty((n, 0))
    else:
        x_centered = x - x.mean(axis=0)

    if sigma0_sq is None:
        sigma0_sq = ols_residual_variance(x_centered, y_centered)

    if p == 0:
        ssr_g = np.dot(y_centered, y_centered)
    else:
        xtx = x_centered.T @ x_centered
        xty = np.einsum('ij,i->j', x_centered, y_centered)
        quad = xty @ solve(xtx, xty, assume_a='sym')
        ssr_g = np.dot(y_centered, y_centered) - (g / (g + 1.0)) * quad

    return (
        -0.5 * n * np.log(np.pi)
        -0.5 * p * np.log(1.0 + g)
        + (nu0 / 2.0) * np.log(nu0 * sigma0_sq)
        - ((nu0 + n) / 2.0) * np.log(nu0 * sigma0_sq + ssr_g)
        + gammaln((nu0 + n) / 2.0)
        - gammaln(nu0 / 2.0)
    )

def posterior_hyperparameters_g_prior(x, y, g=None):
    """
    Posterior under the centered g-prior for the slope coefficients
    """
    n = len(y)
    p = x.shape[1]

    if g is None:
        g = n

    y_centered = y - y.mean()

    if p == 0:
        return {
            'x_bar': np.zeros(0),
            'y_bar': y.mean(),
            'beta_mean': np.zeros(0),
            'beta_cov_base': np.zeros((0, 0)),
            'a_n': n / 2.0,
            'b_n': 0.5 * np.dot(y_centered, y_centered),
        }

    x_bar = x.mean(axis=0)
    x_centered = x - x_bar
    xtx = x_centered.T @ x_centered
    xty = np.einsum('ij,i->j', x_centered, y_centered)
    xtx_inv = np.linalg.inv(xtx)
    beta_hat = solve(xtx, xty, assume_a='sym')

    resid = y_centered - np.einsum('ij,j->i', x_centered, beta_hat)
    sse = np.dot(resid, resid)
    quad = beta_hat @ (xtx @ beta_hat)

    return {
        'x_bar': x_bar,
        'y_bar': y.mean(),
        'beta_mean': (g / (g + 1.0)) * beta_hat,
        'beta_cov_base': (g / (g + 1.0)) * xtx_inv,
        'a_n': n / 2.0,
        'b_n': 0.5 * (sse + quad / (g + 1.0)),
    }

def sample_posterior_for_model(x, y, draws, g=None):
    params = posterior_hyperparameters_g_prior(x, y, g=g)
    sigma2 = invgamma.rvs(a=params['a_n'], scale=params['b_n'], size=draws)

    p = x.shape[1]
    beta_draws = np.zeros((draws, p))

    if p > 0:
        for s in range(draws):
            beta_draws[s] = np.random.multivariate_normal(
                mean=params['beta_mean'],
                cov=sigma2[s] * params['beta_cov_base'],
            )

    alpha_mean = params['y_bar'] - np.einsum('ij,j->i', beta_draws, params['x_bar'])
    alpha_draws = np.random.normal(
        loc=alpha_mean,
        scale=np.sqrt(sigma2 / len(y)),
    )
    return alpha_draws, beta_draws, sigma2

def posterior_mean_fit(x_train, y_train, x_new, included, g=None):
    params = posterior_hyperparameters_g_prior(x_train[:, included], y_train, g=g)
    intercept = params['y_bar']

    if included.any():
        intercept = params['y_bar'] - params['beta_mean'] @ params['x_bar']
        fitted = intercept + np.einsum(
            'ij,j->i',
            x_new[:, included],
            params['beta_mean'],
        )
    else:
        fitted = np.full(len(x_new), intercept)

    return intercept, fitted, params

def model_prior_log_prob(z):
    if use_uniform_model_prior:
        return -len(z) * np.log(2.0)

    z = np.asarray(z)
    return (
        z.sum() * np.log(prior_inclusion_prob)
        + (len(z) - z.sum()) * np.log(1.0 - prior_inclusion_prob)
    )

#---------------------------------------------------------------------
# Load and prepare data
#---------------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.copy()

predictor_names = data.feature_names
outcome_name = 'target'

analysis_df = df[predictor_names + [outcome_name]].dropna().copy()

if n_samples is not None and n_samples < len(analysis_df):
    analysis_df = analysis_df.sample(n=n_samples, random_state=302)

x_df = analysis_df[predictor_names].astype(float)
y_full = analysis_df[outcome_name].to_numpy()

x_train_df, x_test_df, y_train, y_test = train_test_split(
    x_df,
    y_full,
    test_size=test_size,
    random_state=split_random_state,
)

# Match the preprocessing style in linear_regression.py, using only training moments
x_train_mean = x_train_df.mean()
x_train_std = x_train_df.std()
x_train = ((x_train_df - x_train_mean) / x_train_std).to_numpy()
x_test = ((x_test_df - x_train_mean) / x_train_std).to_numpy()

n_train, p = x_train.shape
n_test = x_test.shape[0]
total_n = n_train + n_test
g = n_train
nu0 = 1.0

print('\nBayesian model selection for the diabetes data')
print('Hoff, Section 9.3 setup: modified g-prior with g = n and unit information prior')
print(f'Total sample size: {total_n}')
print(f'Training sample size: {n_train}')
print(f'Test sample size: {n_test}')
print(f"Number of candidate predictors: {p}")
print(f"Total models enumerated: {2**p}")

#---------------------------------------------------------------------
# Enumerate all models exactly
#---------------------------------------------------------------------
model_rows = []

for z in product([0, 1], repeat=p):
    z = np.array(z, dtype=int)
    included = z.astype(bool)
    x_z = x_train[:, included]

    log_py_xz = log_marginal_likelihood_g_prior(
        y=y_train,
        x=x_z,
        g=g,
        nu0=nu0,
    )

    log_prior = model_prior_log_prob(z)

    model_rows.append(
        {
            'z': tuple(z.tolist()),
            'size': int(z.sum()),
            'predictors': [name for name, keep in zip(predictor_names, included) if keep],
            'log_marginal_likelihood': log_py_xz,
            'log_prior': log_prior,
            'log_posterior_kernel': log_prior + log_py_xz,
        }
    )

models_df = pd.DataFrame(model_rows)
log_norm = logsumexp(models_df['log_posterior_kernel'].to_numpy())
models_df['posterior_prob'] = np.exp(models_df['log_posterior_kernel'] - log_norm)
models_df = models_df.sort_values(
    ['posterior_prob', 'log_marginal_likelihood'],
    ascending=False,
).reset_index(drop=True)

#---------------------------------------------------------------------
# Summaries
#---------------------------------------------------------------------
top_models = models_df.head(top_models_to_print).copy()
top_models['model'] = top_models['predictors'].apply(
    lambda x: 'intercept only' if len(x) == 0 else 'intercept + ' + ' + '.join(x)
)

print('\nTop models by posterior probability')
print(
    top_models[['model', 'size', 'posterior_prob']]
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

inclusion_probs = []
for j, name in enumerate(predictor_names):
    post_prob = models_df.loc[models_df['z'].apply(lambda z: z[j] == 1), 'posterior_prob'].sum()
    inclusion_probs.append(
        {
            'predictor': name,
            'posterior_inclusion_prob': post_prob,
        }
    )

inclusion_df = pd.DataFrame(inclusion_probs).sort_values(
    'posterior_inclusion_prob', ascending=False
)

print('\nPosterior inclusion probabilities')
print(inclusion_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_model = models_df.iloc[0]
print('\nHighest posterior probability model')
print(f"z = {best_model['z']}")
print(f"Predictors: {best_model['predictors'] if best_model['predictors'] else ['intercept only']}")
print(f"Posterior probability: {best_model['posterior_prob']:.4f}")

worst_model = models_df.iloc[-1]
print('\nLowest posterior probability model')
print(f"z = {worst_model['z']}")
print(f"Predictors: {worst_model['predictors'] if worst_model['predictors'] else ['intercept only']}")
print(f"Posterior probability: {worst_model['posterior_prob']:.4e}")

#---------------------------------------------------------------------
# Bayesian model averaging for coefficients
#---------------------------------------------------------------------
posterior_mean_beta = np.zeros(p)

for _, row in models_df.iterrows():
    included = np.array(row['z'], dtype=bool)
    if not included.any():
        continue

    params = posterior_hyperparameters_g_prior(x_train[:, included], y_train, g=g)
    posterior_mean_beta[included] += row['posterior_prob'] * params['beta_mean']

bma_df = pd.DataFrame(
    {
        'predictor': predictor_names,
        'posterior_mean_beta_bma': posterior_mean_beta,
    }
).sort_values('posterior_mean_beta_bma', key=lambda s: np.abs(s), ascending=False)

print('\nBayesian model-averaged posterior means for slopes')
print(bma_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

bma_intercept = y_train.mean()
bma_fitted_test = bma_intercept + np.einsum('ij,j->i', x_test, posterior_mean_beta)

#---------------------------------------------------------------------
# Posterior draws from the highest-probability model
#---------------------------------------------------------------------
best_included = np.array(best_model['z'], dtype=bool)
best_predictors = [name for name, keep in zip(predictor_names, best_included) if keep]

best_intercept, best_fitted_test, best_params = posterior_mean_fit(
    x_train,
    y_train,
    x_test,
    best_included,
    g=g,
)

worst_included = np.array(worst_model['z'], dtype=bool)
worst_intercept, worst_fitted_test, worst_params = posterior_mean_fit(
    x_train,
    y_train,
    x_test,
    worst_included,
    g=g,
)

bma_test_mse = np.mean((y_test - bma_fitted_test) ** 2)
best_test_mse = np.mean((y_test - best_fitted_test) ** 2)
worst_test_mse = np.mean((y_test - worst_fitted_test) ** 2)

print('\nTest-set mean squared errors')
print(f'BMA: {bma_test_mse:.4f}')
print(f'Best model: {best_test_mse:.4f}')
print(f'Lowest posterior model: {worst_test_mse:.4f}')

alpha_draws, beta_draws, sigma2_draws = sample_posterior_for_model(
    x_train[:, best_included],
    y_train,
    draws=posterior_draws,
    g=g,
)

coef_rows = [
        {
            'parameter': 'intercept',
            'mean': alpha_draws.mean(),
            'sd': alpha_draws.std(ddof=1),
            'q2.5': np.quantile(alpha_draws, 0.025),
            'median': np.quantile(alpha_draws, 0.5),
            'q97.5': np.quantile(alpha_draws, 0.975),
        }
]

for j, name in enumerate(best_predictors):
    samples = beta_draws[:, j]
    coef_rows.append(
        {
            'parameter': name,
            'mean': samples.mean(),
            'sd': samples.std(ddof=1),
            'q2.5': np.quantile(samples, 0.025),
            'median': np.quantile(samples, 0.5),
            'q97.5': np.quantile(samples, 0.975),
        }
    )

coef_rows.append(
    {
        'parameter': 'sigma',
        'mean': np.sqrt(sigma2_draws).mean(),
        'sd': np.sqrt(sigma2_draws).std(ddof=1),
        'q2.5': np.quantile(np.sqrt(sigma2_draws), 0.025),
        'median': np.quantile(np.sqrt(sigma2_draws), 0.5),
        'q97.5': np.quantile(np.sqrt(sigma2_draws), 0.975),
    }
)

posterior_summary_df = pd.DataFrame(coef_rows)

print('\nPosterior summary for the highest-probability model')
print(posterior_summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

#---------------------------------------------------------------------
# Visualization
#---------------------------------------------------------------------
if show_plots:
    plot_top_models = models_df.head(top_models_to_plot).copy()
    plot_top_models['model_label'] = plot_top_models['predictors'].apply(
        lambda x: 'intercept' if len(x) == 0 else ' + '.join(x)
    )
    plot_top_models = plot_top_models.iloc[::-1]

    inclusion_plot_df = inclusion_df.sort_values(
        'posterior_inclusion_prob', ascending=True
    )

    size_df = (
        models_df.groupby('size', as_index=False)['posterior_prob']
        .sum()
        .sort_values('size')
    )

    fig, axes = plt.subplots(2, 2, figsize=(18,9))

    ax = axes[0, 0]
    ax.barh(
        inclusion_plot_df['predictor'],
        inclusion_plot_df['posterior_inclusion_prob'],
        color='steelblue',
        alpha=0.9,
    )
    ax.axvline(0.5, color='firebrick', linestyle='--', linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('posterior inclusion probability')
    ax.set_title('predictor inclusion probabilities')

    ax = axes[0, 1]
    ax.barh(
        plot_top_models['model_label'],
        plot_top_models['posterior_prob'],
        color='darkseagreen',
        alpha=0.9,
    )
    ax.set_xlabel('posterior probability')
    ax.set_title(f'top {top_models_to_plot} models')

    ax = axes[1, 0]
    ax.bar(
        size_df['size'],
        size_df['posterior_prob'],
        width=0.8,
        color='mediumpurple',
        alpha=0.9,
    )
    ax.set_xticks(size_df['size'])
    ax.set_xlabel('model size')
    ax.set_ylabel('posterior probability')
    ax.set_title('posterior distribution of model size')

    ax = axes[1, 1]
    ax.scatter(y_test, bma_fitted_test, alpha=0.75, s=25, color='dodgerblue', label='bma')
    ax.scatter(y_test, best_fitted_test, alpha=0.75, s=25, color='limegreen', label='best model')
    ax.scatter(y_test, worst_fitted_test, alpha=0.75, s=25, color='orange', label='lowest posterior model')
    lower = min(y_test.min(), bma_fitted_test.min(), best_fitted_test.min(), worst_fitted_test.min())
    upper = max(y_test.max(), bma_fitted_test.max(), best_fitted_test.max(), worst_fitted_test.max())
    ax.plot([lower, upper], [lower, upper], linestyle='--', color='black', label='perfect fit')
    ax.set_xlabel('observed target (test)')
    ax.set_ylabel('fitted value (test)')
    ax.set_title('observed vs fitted on the test set')
    ax.legend()

    plt.tight_layout()
    plt.show()


