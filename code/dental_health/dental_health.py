# Gamma–Poisson illustration: NHANES dental examination data (2013–2014)
"""
Dataset links:
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2013-2014
  Dentition file: OHXDEN_H.XPT
  Demographics file: DEMO_H.XPT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom

#----------------------------------------------------------------
# Setup
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

# plot specifications
k = np.arange(0, 5)
x = np.linspace(0, 2, 500)

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
r1, b1 = alpha_m, beta_m
r2, b2 = alpha_f, beta_f

p1 = b1 / (b1 + 1)
p2 = b2 / (b2 + 1)

fig, axes = plt.subplots(1, 2, figsize=(18,6))

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
