# %%
import pymc3 as pm; print(f"Running on PyMC3 v{pm.__version__}")
from pymc3 import Model, Gamma, Beta, Bernoulli, Exponential, sample
from pymc3.distributions import Interpolated
from scipy import stats
from scipy.special import kl_div as kl_div
import pandas as pd
import numpy as np
import pickle, os, logging
from tqdm import tqdm
from colorama import Fore, Back, Style

cwd = os.getcwd()

class args():
    in_path = os.path.join(cwd, 'data/clean/trials/summarized_pilot_trials.csv')


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


# Load data
df = pd.read_csv(args.in_path, index_col = None)
df = df.assign(success = df.outcome.str.contains('success').astype(int))

# Set up Bayesian models
priors = dict(
    BetaP = dict(alpha=1, beta=1),
    ExponMu = dict(lam=0.001),
    ExponSigma = dict(lam=0.001)
)
nb_samples, tune_size, nb_chains, showprog = 1000, 1000, 2, True
logger = logging.getLogger('pymc3')
logger.propagate = False

exceptions = []
with open(os.path.join(cwd, 'data_processing/temp/exceptions.pkl'), 'wb') as file:
    pickle.dump(exceptions, file)

c1, c2 = Fore.GREEN, Fore.CYAN
color = c1
nb_participants = len(df.participant.unique())
for id, username in enumerate(df.participant.unique()):
    if id <= 31:
        continue
    # try:
    one = df.loc[df.participant.eq(username)]
    traces = {username: {
        'beta0_model': {1: [], 2: [], 3: []}, 
        'wad_model': []
        }
    }
    color = c1 if color == c2 else c2
    print(f'{color}>>> User: {username} ({id+1}/{nb_participants})')
    for sess in [1, 2, 3]:
        if not np.any(one.session==sess):
            print(f'{color}No data for sess {sess}')
            continue
        print(f'{Fore.BLACK}{Back.YELLOW}Session {sess}{Style.RESET_ALL}')
        
        print(f'{color}>>> BETA0 model')
        observed_outcomes = one[one.session==sess].outcome.str.contains('success').to_numpy().astype(int)
        with tqdm(total=observed_outcomes.size, ncols=100) as progbar:
            for oi, o in enumerate(observed_outcomes):
                beta0_model = Model()
                with beta0_model:
                    # Priors from knowledge or from old posterior
                    if sess == 1:
                        if oi == 0:
                            beta0 = Beta('beta0', alpha=priors['BetaP']['alpha'], beta=priors['BetaP']['beta'])
                        else:
                            beta0 = from_posterior(param='beta0', samples=traces[username]['beta0_model'][sess][-1]['beta0'], extrapolate=False)
                    else:
                        sess_ind = sess if oi else sess-1
                        beta0 = from_posterior(param='beta0', samples=traces[username]['beta0_model'][sess_ind][-1]['beta0'], extrapolate=False)
                    # Likelihood (sampling distribution) of observations
                    outcome = Bernoulli('outcome', p=beta0, observed=observed_outcomes)
                    # draw 1000 posterior samples
                    trace = sample(nb_samples, cores=1, chains=nb_chains, tune=tune_size, 
                                   return_inferencedata=False, progressbar=showprog)
                    traces[username]['beta0_model'][sess].append(trace)
                progbar.update(1)
        
        print(f'{color}>>> WAD model')
        observed_outcomes = one[one.session==1].wad.to_numpy()
        wad_model = Model()
        with wad_model:
            # Priors from knowledge or from old posterior
            if sess == 1:
                wad_mu = Exponential('wad_mu', lam=priors['ExponMu']['lam'])
                wad_sigma = Exponential('wad_sigma', lam=priors['ExponSigma']['lam'])
            else:
                wad_mu = from_posterior(param='wad_mu', samples=traces[username]['wad_model'][-1]['wad_mu'], extrapolate=True)
                wad_sigma = from_posterior(param='wad_sigma', samples=traces[username]['wad_model'][-1]['wad_sigma'], extrapolate=True)
            # Likelihood (sampling distribution) of observations
            outcome = Gamma('outcome', mu=wad_mu, sigma=wad_sigma, observed=observed_outcomes)
            # draw 1000 posterior samples
            trace = sample(nb_samples, cores=1, chains=nb_chains, tune=tune_size, return_inferencedata=False, progressbar=showprog)
            traces[username]['wad_model'].append(trace)

    with open(os.path.join(cwd, f'data_processing/temp/{id}.pkl'), 'wb') as file:
        pickle.dump(traces, file)

    # except:
    #     print(f'{Fore.RED}{Back.YELLOW} EXCEPTION ON USER {username}{Style.RESET_ALL}')
    #     exceptions.append((id, username))
    #     with open(os.path.join(cwd, f'data_processing/temp/{id}.pkl'), 'wb') as file:
    #         pickle.dump(exceptions, file)

print(Style.RESET_ALL)
print(traces)
# %%

# %%

'''
Now write some code to integrate the model data with summarize sessions
    - Need to take each participant's traces and compute a DKL between
        - Prior and posterior after sess 1, 2, and 3 (JOLD 1 for sessions 1, 2, and 3)
        - Prior and posterior after sess 1 (JOLD 2 sess 1)
            - Look at how well JOLD1 and JOLD2 questions are correlated
            - Also do posterior after sess 1 and posterior after sess 3 (JOLD 2 sess 3)
        - Prior and posterior after sess 3(JOLD3)
Fit models:
    - JOLD 1 vs DKLs
    - JOLD 1 and baselines
Read Maxime's report and write some feedback
'''