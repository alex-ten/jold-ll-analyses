#region: Imports and functions
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display as disp

import pymc3 as pm; print(f"Running on PyMC3 v{pm.__version__}")
from pymc3 import Model, Gamma, Beta, Bernoulli, Exponential, sample, traceplot
from pymc3.distributions import Interpolated
from scipy import stats
from scipy.special import kl_div as kl_div

import distcan


def convert_beta_params(mean, var):
    alpha = mean**2 * ((1-mean)/var - 1/mean)
    beta = alpha * (1/mean - 1)
    return alpha, beta


def beta_dkl(x, params1, params2):
    sample1 = stats.beta.pdf(x, a=params1[0], b=params1[1])
    sample1 = sample1 / sample1/np.sum(sample1)
    
    sample2 = stats.beta.pdf(x, a=params2[0], b=params2[1])
    sample2 = sample2 / sample2/np.sum(sample2)
    
    print(kl_div(sample1, sample2))

    return None


def nonzero_probs(sample):
    fake_zero = sample[sample > 0].min()
    out = sample.copy()
    out[sample == 0] = fake_zero
    return out


def from_posterior(param, samples, extrapolate=True):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    if extrapolate:
        x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)
#endregion

#region: Load data
#%%
in_path = '../data/clean/trials/summarized_pilot_trials.csv'

df = pd.read_csv(in_path)
participants = df.participant.unique()
username = participants[4]
one = df.loc[df.participant.eq(username)]

print(f'Username: {username}')
disp(one.head())
#endregion

#region: Baseline competence update
# %%
# PyMC3 example of Bayesian updating of success probability in 3 sessions
prior = dict(alpha=1, beta=1)
traces = []
print('Model 1')
for sess in [1]:
    if not np.any(one.session==sess):
        continue
    print(f'Session {sess}')
    observed_outcomes = one[one.session==sess].outcome.str.contains('success').to_numpy().astype(int)
    print(f'Performance: {observed_outcomes}')
    beta0_model = Model()
    with beta0_model:

        # Priors from knowledge or from old posterior
        if sess == 1:
            beta0 = Beta('beta0', alpha=prior['alpha'], beta=prior['beta'])
        else:
            beta0 = from_posterior(param='beta0', samples=traces[-1]['beta0'], extrapolate=False)
            
        # Likelihood (sampling distribution) of observations
        outcome = Bernoulli('outcome', p=beta0, observed=observed_outcomes)

        # draw 1000 posterior samples
        trace = sample(1000, cores=1, return_inferencedata=False)
        traces.append(trace)
#endregion

#region: Baseline competence update visualization
#%%
plt.close('all')
fig, axes = plt.subplots(figsize=[4, 4*len(traces)], ncols=1, nrows=max(2, len(traces)))
for i, trace in enumerate(traces):
    samples = trace['beta0']
    smin, smax = np.min(samples), np.max(samples)
    x = np.linspace(smin, smax, 100)

    # Plot prior
    if i == 0:
        y_prior = stats.beta.pdf(x, a=prior['alpha'], b=prior['beta'])
    else:
        y_prior = y_post
    axes[i].plot(x, y_prior, color='gray', label='Prior')
    axes[i].set_ylabel('Probability')

    # Plot sample from posterior
    y_post = stats.gaussian_kde(samples)(x)
    sample_stats = (samples.mean(), samples.std())
    axes[i].plot(x, y_post, color='orange', label=f'Post (M={sample_stats[0]:.3f}, SD={sample_stats[1]:.3f})')

    axes[i].legend()

    # Calculate KL divergence
    p = y_prior / y_prior.sum()
    q = y_post / y_post.sum()
    dkl = kl_div(p, q)
    axes[i].set_title(f'Session {i+1} | DKL = {dkl.sum():.3f}')
#endregion

#region: WAD update
# %%
# PyMC3 example of Bayesian updating of WAD
observed_outcomes = one[one.session==1].wad.to_numpy()
# observed_outcomes = observed_outcomes - observed_outcomes.mean()

prior = dict(mu_wad=0.001, sigma_wad=0.001)
prior['shape'] = prior['mu_wad']**2 / prior['sigma_wad']**2
prior['scale'] = prior['sigma_wad']**2 / prior['mu_wad']

traces = []
print('Model 2')
for sess in [1]:
    if not np.any(one.session==sess):
        continue
    wad_model = Model()
    with wad_model:

        # Priors from knowledge or from old posterior
        if sess == 1:
            mu_wad = Exponential('mu_wad', lam=prior['mu_wad'])
            sigma_wad = Exponential('sigma_wad', lam=prior['sigma_wad'])
        else:
            mu_wad = from_posterior(param='mu_wad', samples=traces[-1]['mu_wad'])
            sigma_wad = from_posterior(param='sigma_wad', samples=traces[-1]['sigma_wad'])

        # Likelihood (sampling distribution) of observations
        outcome = Gamma('outcome', mu=mu_wad, sigma=sigma_wad, observed=observed_outcomes)
        # draw 1000 posterior samples
        trace = sample(1000, cores=2, return_inferencedata=False)
        traces.append(trace)
#endregion

#region: WAD updates visualization
# %%
nsamples = 1000
plt.close('all')
fig, axes = plt.subplots(nrows=len(traces), ncols=1, figsize=[4, 12])
for i, trace in enumerate(traces):
    # Sample WADs from prior and plot
    if i == 0:
        prior_mu = stats.expon(scale=1/prior['mu_wad']).rvs(nsamples)
        prior_sigma = stats.expon(scale=1/prior['sigma_wad']).rvs(nsamples)
        prior_shape = prior_mu**2 / prior_sigma**2
        prior_scale = prior_sigma**2 / prior_mu
        wad_prior = np.concatenate(
            [distcan.Gamma(alpha=prior_shape[i], beta=prior_scale[i]).rvs(size=1) for i in range(nsamples)]
        )
    else:
        wad_prior = wad_post
    sample_stats = wad_prior.mean(), wad_prior.var()
    sns.histplot(x=wad_prior, stat='density', kde=True, ax=axes[i], element='step', color='gray')

    # Sample WADs from posterior and plot
    mu_sample = trace['mu_wad']
    sigma_sample = trace['sigma_wad']
    mus = np.linspace(np.min(mu_sample), np.max(mu_sample), nsamples)
    sigmas = np.linspace(np.min(sigma_sample), np.max(sigma_sample), nsamples)
    post_shape = mus**2 / sigmas**2
    post_scale = sigmas**2 / mus
    wad_post = np.concatenate(
        [distcan.Gamma(alpha=post_shape[i], beta=post_scale[i]).rvs(size=1) for i in range(nsamples)]
    )

    sample_stats = wad_post.mean(), wad_post.var()
    sns.histplot(x=wad_post, stat='density', kde=True, ax=axes[i], element='step', color='orange')

    axes[i].set_xlim(wad_post.min(), wad_post.max())

    p = wad_prior / wad_prior.sum()
    q = wad_post / wad_post.sum()
    dkl = kl_div(p, q)
    axes[i].set_title(f'Session {i+1}; DKL = {dkl.sum():.3f}')

# Plot WAD hyperparameter update
hparams = ['mu_wad', 'sigma_wad']
fig, axes = plt.subplots(nrows=len(traces), ncols=len(hparams), figsize=[8, 12])
for i, trace in enumerate(traces):
    for j, param in enumerate(hparams):
        axes[i, j].set_ylabel('Probability')
        axes[i, j].set_title(param)
        
        samples = trace[param]
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        
            # plot prior
        if i == 0:
            y_prior = stats.expon.pdf(x, loc=0, scale=1/prior[param])
        else:
            y_prior = y_post
        axes[i, j].plot(x, y_prior, color='gray')

        # plot posterior
        y_post = stats.gaussian_kde(samples)(x)
        axes[i, j].plot(x, y_post, color='orange')

        p = y_prior / y_prior.sum()
        q = y_post / y_post.sum()
        dkl = kl_div(p, q)
        axes[i, j].set_title(f'Session {i+1}; {param}: DKL = {dkl.sum():.3f}')

#endregion

# %%
