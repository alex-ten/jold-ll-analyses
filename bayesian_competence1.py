# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, beta, binom
import scipy.special as spec
from bayespy.inference import VB
from bayespy.nodes import Bernoulli, Beta, Binomial

#region: DEFINE SOME FUNCTIONS
# %% 
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


def betaDKL(f, g):
    '''Computes the KL divergence between two beta distributions parameterized by 
    parameter paire f and g, respectively. More precisely, the function returns
    $D_{KL}(F, G)$, where $F \sim \mathrm{Beta}(f_0, f_1)$ and $G \sim \mathrm{Beta}(g_0, g_1)$.
    The formula is taken from: https://math.wikia.org/wiki/Beta_distribution

    Parameters
    ----------
    f : numpy.array
        An array of size 2 that contains the $\alpha$ and $\beta$ parameters of the 1st distribution
    g : numpy.array
        An array of size 2 that contains the $\alpha$ and $\beta$ parameters of the 2nd distribution

    Returns
    -------
    float
        The KL divergence of two beta distributions.
    '''
    a = np.log(spec.beta(g[0], g[1])/spec.beta(f[0], f[1]))
    b = -(g[0]-f[0])*spec.digamma(f[0]) - (g[1]-f[1])*spec.digamma(f[1])
    c = (g[0]-f[0]+g[1]-f[1])*spec.digamma(f.sum())
    return a + b + c


def betaPlot(ax, params, **plot_kwargs):
    comp_space = np.linspace(0, 1, 500)
    distrib = beta(*params)
    x = comp_space
    y = distrib.pdf(comp_space)
    ax.plot(x, y, **plot_kwargs)


def betaPlot2(ax, params, **plot_kwargs):
    comp_space = np.linspace(0, 1, 500)
    mu, kappa = params
    a = mu*(kappa - 1)
    b = (kappa - 1)*(1 - mu)
    distrib = beta(a, b)
    x = comp_space
    y = distrib.pdf(comp_space)
    ax.plot(x, y, **plot_kwargs)
#endregion

#region: CREATE LEARNERS
# %% 
T = 30
t = np.arange(T)
N = 5

alpha_space = np.logspace(-3, 0.1, N).round(2)
p0, p1 = .01, .70

plt.close('all')
plt.figure(num='Competence curves')
for i, alpha in enumerate(alpha_space):
    x = t
    y = competence_curve(t, alpha, p0, p1)
    plt.gca().plot(x, y, color=plt.cm.Set1(i), label=f'{alpha:.2f}', lw=2)

plt.ylim(0, 1)
plt.legend(bbox_to_anchor=[1.05, .5], loc='center left', title='Learning rate')

# Sample trials with binary outcomes
samples = []
for alpha in alpha_space:
    competence = competence_curve(t, alpha, p0, p1)
    samples.append(bernoulli.rvs(competence, size=T))
samples = np.array(samples)

print(samples)
#endregion

#region: CREATE PRIORS
# %% 
comp_space = np.linspace(0, 1, 500)
priors = {
    'unconfident' : np.array([.1, 99.9]),
    'neutral' : np.array([1, 1]),
    'confident': np.array([2, 1]),
    # 'overconfident' : np.array([0.8, 0.2]),
}
alt_priors = {
    'unconfident' : np.array([0.01, 100]),
    'neutral' : np.array([.5, 5]),
    'optimistic' : np.array([.5, 10]),
    # 'overconfident' : np.array([0.8, 0.2]),
}


plt.close('all')
plt.figure('Priors')
for i, (k, v) in enumerate(priors.items()):
    betaPlot(plt.gca(), v, color=plt.cm.Set2(i), label=f'{k.capitalize()} {v}', lw=2)

plt.gca().set_ylim(0, 8)
plt.gca().set_xlim(0, 1)
plt.gca().legend(bbox_to_anchor=[.5, .9], loc='upper center', title='Priors')   
#endregion

#region: SIMULATE TRIALS
# %% 
sampling = 'Bernoulli' # or either 'Bernoulli' or 'Binomial'
n_binom = 5 # Sets the parameter n for Binomial sampling model

surprise_data = {}
posterior_data = {}
for k, v in priors.items():
    surprise_data[k] = []
    posterior_data[f'{k}'] = []
    for sample in samples:
        surprise_history = []
        posterior_history = []
        prior = priors[k]
        for o in sample:
            update, surprise, posterior = False, np.nan, np.array([np.nan, np.nan])
            competence_prior = Beta(alpha=prior) # alpha is a pair of Beta distribution parameters (a, b)
            if sampling.lower() == 'bernoulli':
                sampling_model = Bernoulli(p=competence_prior, name='outcome', plates=(1,))
                sampling_model.observe(o[np.newaxis])
            elif sampling.lower() == 'binomial':
                sampling_model = Binomial(n=n_binom, p=competence_prior, plates=(1,))
                sampling_model.observe(o.sum()[np.newaxis])
            Q = VB(competence_prior, sampling_model)
            Q.update(repeat=20, verbose=False)
            posterior = competence_prior.get_parameters()[0]
            surprise = betaDKL(posterior, prior)
            prior = posterior
            
            surprise_history.append(surprise)
            posterior_history.append(posterior)

        surprise_data[k].append(np.array(surprise_history))
        posterior_data[f'{k}'].append(np.stack(posterior_history, axis=0))

    surprise_data[k] = np.stack(surprise_data[k], axis=0)
    posterior_data[f'{k}'] = np.stack(posterior_data[f'{k}'])
#endregion

#region: VISUALIZE SURPRISE
# %%
plt.close('all')
fig, ax = plt.subplots(num='Surprise', nrows=len(priors.keys()), ncols=2, figsize=[10, 7.5])

for i, (k, v) in enumerate(surprise_data.items()):
    ax[i, 0].set_title(f'{k.capitalize()}: Beta{priors[k]}')
    for j, hist in enumerate(v):
        y = hist*samples[j]
        # print(y)
        x = np.arange(y.size)
        ax[i, 0].plot(x, y, label=f'{alpha_space[j]:.2f}', color=plt.cm.Set1(j), marker='o')
    ax[i, 0].legend()
    x = np.arange(v.shape[0])
    y = np.mean(v*samples, axis=1)
    ax[i, 1].bar(x, y, color=[plt.cm.Set1(j) for j, _ in enumerate(v)])

fig.tight_layout()
#endregion

#region: POSTERIOR EVOLUTION
# %%
plt.close('all')
plt.figure(figsize=[6,10])
plt.suptitle(f'Posterior distributions')
color_nums = np.linspace(0, T)
for i, (k, v) in enumerate(posterior_data.items()):
    if k == 'neutral':
        for j, hist in enumerate(v):
            plt.subplot(5, 1, j+1)
            betaPlot(plt.gca(), priors[k], color='k', label=priors[k], lw=2, ls='--')
            color_space = np.linspace(0, T)
            for pi, params in enumerate(hist):
                label = f'{params}' if (pi==0) or (pi==T-1) else None
                betaPlot(plt.gca(), params, color=plt.cm.viridis(color_space[pi]), lw=2, label=label, alpha=.5)

            plt.legend()
fig.tight_layout()
#endregion

# %%
