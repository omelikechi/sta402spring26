# Gamma–Poisson illustration: NHANES dental examination data (2013–2014)
"""
Dataset links:
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2013-2014
  - Dentition file: OHXDEN_H.XPT
  - Demographics file: DEMO_H.XPT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom

#----------------------------------------------------------------
# Setup
#----------------------------------------------------------------
np.random.seed(302)

show_summary = False
show_posterior_prediction = False
show_model_checking = True

#----------------------------------------------------------------
# Data
#----------------------------------------------------------------
"""
Tooth condition codes used in NHANES dentition data:
  D = Decayed tooth (active disease)
  R = Restored tooth (filled/treated)
  M = Missing tooth
"""
codes_to_count = ['D', 'M', 'R']
outcome_name = 'count_' + ''.join(codes_to_count)

age_min = 50
age_max = 50

alpha_prior = 1
beta_prior = 1

#----------------------------------------------------------------
# Load data
#----------------------------------------------------------------
df = pd.read_sas('OHXDEN_H.XPT')
ctc_vars = [c for c in df.columns if c.endswith('CTC')]

# Convert SAS byte strings to normal strings
for c in ctc_vars:
	df[c] = df[c].map(lambda x: x.decode() if isinstance(x, bytes) else x)

# Compute tooth-count outcome once
df[outcome_name] = df[ctc_vars].isin(codes_to_count).sum(axis=1)

# Merge demographics (sex + age)
demo = pd.read_sas('DEMO_H.XPT')[['SEQN','RIAGENDR','RIDAGEYR']]
df = df.merge(demo, on='SEQN')

# Optional age restriction
if age_min is not None:
	df = df[df['RIDAGEYR'] >= age_min]
if age_max is not None:
	df = df[df['RIDAGEYR'] <= age_max]

# Split groups
male_counts = df.loc[df['RIAGENDR'] == 1.0, outcome_name].values
female_counts = df.loc[df['RIAGENDR'] == 2.0, outcome_name].values

# Counts and size for each group
sum_male, sum_female = male_counts.sum(), female_counts.sum()
n_male, n_female = len(male_counts), len(female_counts)

# Basic summary
if show_summary:
	summary = pd.DataFrame({
		'Total': [n_male, n_female],
		'Count': [sum_male, sum_female],
		'Mean': [male_counts.mean(), female_counts.mean()],
		'Variance': [male_counts.var(ddof=1), female_counts.var(ddof=1)]
	}, index=['Male', 'Female'])

	print(summary.to_string(formatters={'Mean': lambda x: f'{x:.3f}', 'Variance': lambda x: f'{x:.3f}'}))
	print()

#----------------------------------------------------------------
# Gamma-Poisson
#----------------------------------------------------------------
"""
theta ~ Gamma(alpha, beta) and y_i | theta ~ Poisson(theta), then the posterior is
theta | y_1,...,y_n ~ Gamma(a + sum_{i=1}^n y_i, b + n)
"""

alpha_m = alpha_prior + sum_male
beta_m = beta_prior + n_male
alpha_f = alpha_prior + sum_female
beta_f = beta_prior + n_female

if show_summary:
	print(f'Prior mean: {alpha_prior / beta_prior:.3f}')
	print(f'Posterior mean (Male): {alpha_m / beta_m:.3f}')
	print(f'Posterior mean (Female): {alpha_f / beta_f:.3f}')

	# Monte Carlo comparison
	S = 100000
	theta_samples_m = np.random.gamma(alpha_m, 1/beta_m, S)
	theta_samples_f = np.random.gamma(alpha_f, 1/beta_f, S)
	print('\nP(theta_female > theta_male | y_1,...,y_n) =', np.mean(theta_samples_f > theta_samples_m))

#----------------------------------------------------------------
# Posterior prediction
#----------------------------------------------------------------
if show_posterior_prediction:

	r1, b1 = alpha_m, beta_m
	r2, b2 = alpha_f, beta_f

	p1 = b1 / (b1 + 1)
	p2 = b2 / (b2 + 1)

	fig, axes = plt.subplots(1, 2, figsize=(18,6))

	# Plot specifications
	k = np.arange(0, 5)
	x = np.linspace(0, 2, 500)

	# Posterior predictive (counts)
	width = 0.25
	axes[0].bar(k - width/2, nbinom.pmf(k, r1, p1), width=width, label='Male')
	axes[0].bar(k + width/2, nbinom.pmf(k, r2, p2), width=width, label='Female')
	axes[0].set_title(r'$p(\tilde{y}\mid y_1,\dots,y_n)$', fontsize=22)
	axes[0].set_xlabel('Unhealthy tooth count', fontsize=22)
	axes[0].set_ylabel('Probability', fontsize=22)
	axes[0].legend()

	# Posterior distributions of Poisson rates
	axes[1].plot(x, gamma.pdf(x, alpha_m, scale=1/beta_m), label='Male', lw=3)
	axes[1].plot(x, gamma.pdf(x, alpha_f, scale=1/beta_f), label='Female', lw=3)
	axes[1].set_title(r'$p(\theta \mid y_1,\dots,y_n)$', fontsize=24)
	axes[1].set_xlabel(f'$\\theta$', fontsize=22)
	axes[1].set_ylabel('Density', fontsize=22)
	axes[1].legend()

	plt.tight_layout()
	plt.show()

#----------------------------------------------------------------
# Posterior predictive model checking
#----------------------------------------------------------------
if show_model_checking:
	group_name = 'women'

	if group_name == 'men':
		n = n_male
		y = male_counts
		alpha = alpha_m
		beta = beta_m
	else:
		n = n_female
		y = female_counts
		alpha = alpha_f
		beta = beta_f

	print(f'\nNumber of {group_name} in age group: {n}')
	print(f'Number of unhealthy teeth: {sum(y)}')
	print(f'Sample mean: {sum(y) / n:.2f}')
	print(f'Sample variance: {np.var(y):.2f}')

	print(f'\nHow do sample mean and variance compare? What does this say about using a Poisson model?')

	# Empirical distribution
	k_emp = np.arange(0, max(y) + 1)
	emp_pmf = np.array([(y == val).mean() for val in k_emp])

	# Posterior predictive distribution
	p = beta / (beta + 1)
	pp_pmf = nbinom.pmf(k_emp, alpha, p)

	# Plot empirical vs posterior predictive distribution
	plt.figure(figsize=(16,8))

	plt.bar(k_emp - 0.2, emp_pmf, width=0.4, label='Empirical')
	plt.bar(k_emp + 0.2, pp_pmf, width=0.4, label='Posterior predictive')

	# plt.xlim(-1, 11)
	plt.xlabel('Unhealthy tooth count', fontsize=18)
	plt.ylabel('Probability', fontsize=18)
	if age_min == age_max:
		plt.title(f'Empirical vs posterior predictive ({group_name} aged {age_min})', fontsize=24)
	elif age_max > 90:
		plt.title(f'Empirical vs posterior predictive ({group_name} over {age_min})', fontsize=24)
	else:
		plt.title(f'Empirical vs posterior predictive ({group_name} ages {age_min} to {age_max})', fontsize=24)
	plt.legend(fontsize=20)
	plt.tight_layout()
	plt.show()

	# Posterior predictive check: Proportion with n_unhealthy unhealthy teeth
	S = 100000
	n_unhealthy = 0
	theta = np.random.gamma(alpha, 1/beta, S)
	pp_prop = np.empty(S)
	for i in range(S):
		y_data = np.random.poisson(theta[i], size=n)
		pp_prop[i] = np.mean(y_data == n_unhealthy)

	empirical_prop = np.mean(y == n_unhealthy)

	# Two-sided posterior predictive p-value
	proportion_greater = np.mean(pp_prop >= empirical_prop)
	proportion_less = np.mean(pp_prop <= empirical_prop)
	print(f'\nObserved proportion with {n_unhealthy} unhealthy teeth: {empirical_prop:.3f}')
	print(f'Posterior predictive proportion with {n_unhealthy} unhealthy teeth: {pp_prop.mean():.3f}')
	print(f'Proportion of simulated datasets MORE extreme than observed data: {proportion_greater:.5f}')
	print(f'Proportion of simulated datasets LESS extreme than observed data: {proportion_less:.5f}')

	plt.figure(figsize=(16,8))
	plt.hist(pp_prop, color='deepskyblue', edgecolor='black', bins=15, density=True)
	plt.axvline(empirical_prop, color='red', lw=6, linestyle='--', label='Observed value')
	plt.title(f'PPD: Proportion of individuals with {n_unhealthy} unhealthy teeth (S = {S})', fontsize=24)
	plt.legend(fontsize=22)
	plt.tight_layout()
	plt.show()







