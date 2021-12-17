# %%
import pymc3 as pm; print(f"Running on PyMC3 v{pm.__version__}")
from pymc3.distributions import Interpolated
from scipy import stats
from scipy.special import rel_entr
import pandas as pd
import numpy as np
import statsmodels.formula.api as sma
from IPython.display import display as disp
import pickle, os
cwd = os.getcwd()
print(f'Current working directory: {cwd}')
# %%

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


def get_pickled_data(id):
    with open(os.path.join(os.getcwd(), f'../data_processing/temp/{id}.pkl'), 'rb') as file:
        data = pickle.load(file)
    return data


def get_linspace_sample(size, trace_sample):
    return np.linspace(np.min(trace_sample), np.max(trace_sample), size)


def bayesian_surprise(prior, posterior):
    p = prior / prior.sum()
    q = posterior / posterior.sum()
    DKL = np.sum(rel_entr(p, q))
    return DKL

def bayesian_surprise_by_trial():
    pass

class args():
    in_path = os.path.join(cwd, '../data/clean/trials/summarized_pilot_sessions1.csv')
    out_path = os.path.join(cwd, '../data/clean/trials/summarized_pilot_sessions2.csv')
    outcomes_data = os.path.join(cwd, '../data/clean/trials/summarized_pilot_trials.csv')
#%%


# Load data
df = pd.read_csv(args.in_path, index_col = None)
df = df.replace(
    {'id':  dict([(username, id) for id, username in enumerate(df.participant.unique())])}
)
outcomes = pd.read_csv(args.outcomes_data, index_col=None).filter(items=['participant', 'session', 'forced', 'outcome'])
outcomes = outcomes.assign(outcome = (outcomes.outcome == 'success').astype(int))
outcomes = outcomes.set_index(['participant', 'session', 'forced'])
# Prior parameters (make sure it's the same as in run_models.py)
priors = dict(
    BetaP = dict(alpha=1, beta=1),
    ExponMu = dict(lam=0.001),
    ExponSigma = dict(lam=0.001)
)
# %%

# Model DKLs (Bayesian surprise)
keys = ['participant', 'session', 'surpBeta0', 'surpWadMu', 'surpWadSigma']
model_data = dict(list(zip(keys, [[] for key in keys])))
for filename in os.listdir('temp/'):
    if 'exceptions' in filename:
        continue
    id = int(filename.split('.')[0])
    traces = get_pickled_data(id)
    username = list(traces.keys()).pop()
    traces = traces[username]
    beta0_traces_dict = traces['beta0_model']
    wad_traces_list = traces['wad_model']
    print(username)
    if sum([bool(l) for l in beta0_traces_dict.values()]) == len(wad_traces_list):
        for i, (t_beta0, t_wad) in enumerate(zip(beta0_traces_dict.values(), wad_traces_list)):
            # Sample from posteriors
            sample_beta0_list = [t['beta0'] for t in t_beta0]
            sample_wad_mu = t_wad['wad_mu']
            sample_wad_sigma = t_wad['wad_sigma']

            # Generate linearly-spaced samples based on posterior samples
            x_beta0_list = [get_linspace_sample(100, s) for s in sample_beta0_list]
            x_wad_mu = get_linspace_sample(100, sample_wad_mu)
            x_wad_sigma = get_linspace_sample(100, sample_wad_sigma)
            
            # Generate prior samples either from prior parameters or from posterior samples from the past
            if i == 0:
                y_prior_beta0 = stats.beta.pdf(x_beta0_list[0], a=priors['BetaP']['alpha'], b=priors['BetaP']['beta'])
                y_prior_wad_mu = stats.expon.pdf(x_wad_mu, loc=0, scale=1/priors['ExponMu']['lam'])
                y_prior_wad_sigma = stats.expon.pdf(x_wad_sigma, loc=0, scale=1/priors['ExponSigma']['lam'])
            else:
                y_prior_beta0 = trial_by_trial_beta0_samples[-1]
                y_prior_wad_mu = y_prior_wad_mu
                y_prior_wad_sigma = y_prior_wad_sigma

            # Estimate posterior PDF
            trial_by_trial_beta0_samples = [y_prior_beta0]+[stats.gaussian_kde(s)(x) for s, x in zip(sample_beta0_list, x_beta0_list)]
            trial_by_trial_beta0_surprises = []
            outs = outcomes.loc[username, i+1, True].values.squeeze()
            for o, y0, y1 in zip(outs, trial_by_trial_beta0_samples[:-1], trial_by_trial_beta0_samples[1:]):
                surprise = bayesian_surprise(y0, y1)
                trial_by_trial_beta0_surprises.append(
                    surprise if o else -surprise
                )
            trial_by_trial_beta0_surprises = np.array(trial_by_trial_beta0_surprises)
            y_post_wad_mu = stats.gaussian_kde(sample_wad_mu)(x_wad_mu)
            y_post_wad_sigma = stats.gaussian_kde(sample_wad_sigma)(x_wad_sigma)

            # Calculate DKL(prior || posterior) and store data
            model_data['participant'].append(username)
            model_data['session'].append(i+1)
            model_data['surpBeta0'].append(np.sum(trial_by_trial_beta0_surprises))
            model_data['surpWadMu'].append(bayesian_surprise(y_prior_wad_mu, y_post_wad_mu))
            model_data['surpWadSigma'].append(bayesian_surprise(y_prior_wad_sigma, y_post_wad_sigma))
    else:
        print('SOMETHING IS WRONG')

model_data_df = pd.DataFrame(model_data)
df = df.merge(model_data_df, on=['participant', 'session'])

# Save dataset
print(f'Saving data to {args.out_path}')
df.to_csv(
    path_or_buf = args.out_path,
    sep = ',',
    index = False
)
# %%
