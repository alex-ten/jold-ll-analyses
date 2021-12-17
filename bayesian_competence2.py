# %% IMPORTS
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

from bayespy.inference import VB
import bayespy.nodes as nodes
import bayespy.plot as bpplt


#region: FUNCTIONS
# %% 
class args():
    in_path = './data/clean/trials/summarized_pilot_trials.csv'
    out_path = './data/clean/trials/summarized_pilot_sessions.csv'


def logistic(x):
    return exp(x) / (1 + exp(x))


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def competence_curve(t, alpha, p0, p1):
    '''Simple concave curve with diminishing returns. This function assumes that uptake is constant across 
    feedback type (positive or negative), so this uptake function only depends on time indexed by x.

        Parameters
        ----------
        x : a numpy.float (or float)
            temporal index
        alpha : float
            uptake rate
        p0 : float
            initial rate
        p1 : float
            asymptotic rate

    Returns
    -------
    same type as x
        uptake rate at x
    '''     
    return p1 - (p1 - p0) * np.exp(-alpha*t)


def plot_norm(ax, x, params, **plot_kwargs):
    y = norm.pdf(x, *params)
    ax.plot(x, y, **plot_kwargs)


def gamma_plot(ax, params, **plot_kwargs):
    comp_space = np.linspace(0, 1000)
    distrib = norm(*params)
    x = comp_space
    y = distrib.pdf(comp_space)
    ax.plot(x, y, **plot_kwargs)


def normDKL(m0, m1, s0, s1):
    return ((s0/s1)**2 + ((m1-m0)**2)/(s1**2) - 1 + 2*np.log(s1/s0))/2

#endregion

#region: GET SOME EXAMPLE DATA
# %% 
df = pd.read_csv(args.in_path, index_col = None)
df = df.assign(success = df.outcome.str.contains('success').astype(int))

# Sort values
df = df.set_index(['participant','session','forced'])
df = df.sort_values(by=['participant','session','forced','trial'])

# Calculate last-5 trials success rate
df = df.assign(
    sr = df.groupby(['participant', 'session', 'forced'])[['success']].rolling(min_periods=1, window=5).mean()
)

ps = df.index.unique(level=0)
id_to_sample = ps[12]
data = df.loc[(id_to_sample, 1, True), ('trial', 'wad', 'success', 'time_trial', 'presses')]
#endregion

#region: BAYESIAN UPDATE OF NORMAL WITH UNKNOWN MU AND SIGMA
# %%
mysample = data.loc[:, 'wad'].to_numpy()

fig, ax = plt.subplots()
N = data.shape[0]
xx = np.linspace(0, 1000, 500)

mu = np.mean(mysample[:2])
sigma = np.std(mysample[:2])
v = 1
b = 1

plot_norm(ax, xx, [mu, sigma], color='k', lw=3, alpha=1)

for i, o in enumerate(mysample[2:]):
    mu_, sigma_ = mu, sigma
    b = b + 0.5 * v / (v + 1) * (o - mu)**2    
    mu = (v*mu + o) / (v + 1)
    v += 1
    sigma = (2*b*(v+1) / (v*v))**0.5
    dkl = normDKL(mu_, mu, sigma, sigma_)
    print(mu, sigma, dkl)
    plot_norm(ax, xx, [mu, sigma], color=mpl.cm.brg(1 - i / N), lw=1, alpha=.5)
    

fig.tight_layout()
#endregion

#region: MLE ESTIMATES OF MU AND SIGMA
# %%
data = df.loc[(id_to_sample, 1, True), ('trial', 'wad', 'success', 'time_trial', 'presses')]
mysample = data.loc[:, 'wad'].to_numpy()

fig, ax = plt.subplots()
N = data.shape[0]
xx = np.linspace(0, 1000, 500)

mu = np.mean(mysample[:2])
sigma = np.std(mysample[:2])

plot_norm(ax, xx, [mu, sigma], color='k', lw=3, alpha=1)

surprise_history = []
for n, o in enumerate(mysample[2:], 3):
    mu_, sigma_ = mu, sigma
    mu = (o + (n - 1) * mu) / n
    sigma = np.sqrt(((n - 2)/(n - 1)) * sigma**2 + (o - mu)**2 / n)
    dkl = normDKL(mu_, mu, sigma, sigma_)
    signed_surprise = dkl if mu > mu_ else -dkl
    surprise_history.append(signed_surprise)
    plot_norm(ax, xx, [mu, sigma], color=mpl.cm.brg(1 - n / (N-2)), lw=1, alpha=.5)
surprise_history = np.array(surprise_history)
ax.axvline(mu)

surprise_history = surprise_history[surprise_history>0]
fig, ax = plt.subplots(num='surp')
ax.plot((surprise_history))
ax.axhline((surprise_history).mean())
fig.tight_layout()
N_old = N
#endregion

# %%
data = df.loc[(id_to_sample, 2, True), ('trial', 'wad', 'success', 'time_trial', 'presses')]
mysample = data.loc[:, 'wad'].to_numpy()

fig, ax = plt.subplots()
N = data.shape[0]
xx = np.linspace(0, 1000, 500)

plot_norm(ax, xx, [mu, sigma], color='k', lw=3, alpha=1)

surprise_history = []
for n, o in enumerate(mysample[:], 2+N_old):
    mu_, sigma_ = mu, sigma
    mu = (o + (n - 1) * mu) / n
    sigma = np.sqrt(((n - 2)/(n - 1)) * sigma**2 + (o - mu)**2 / n)
    dkl = normDKL(mu_, mu, sigma, sigma_)
    signed_surprise = dkl if mu > mu_ else -dkl
    surprise_history.append(signed_surprise)
    plot_norm(ax, xx, [mu, sigma], color=mpl.cm.brg(1 - n / (N-2)), lw=1, alpha=.5)
surprise_history = np.array(surprise_history)
ax.axvline(mu)

surprise_history = surprise_history[surprise_history>0]
fig, ax = plt.subplots(num='surp')
ax.plot((surprise_history))
ax.axhline((surprise_history).mean())
fig.tight_layout()
# %%
