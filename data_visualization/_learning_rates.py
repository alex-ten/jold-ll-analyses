#%% 
import pandas as pd
from IPython.display import display as disp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('seaborn-whitegrid')
from statsmodels.formula.api import logit
import seaborn as sns
from scipy.stats import zscore

#%%
df1 = pd.read_csv('../data/clean/trials/clean_pilot_trials.csv')
df1['init_dist_cond'] = np.abs(df1.init_site - df1.plat_site).astype(str)
df2 = pd.read_csv('../data/prof_data.csv')

df = df1.merge(df2, left_on='participant', right_on='user', how='right')
df['time_cond'] = df.tenmin.astype(str).replace({'True': '10 min', 'False': '20 min'})
df['success'] = df.outcome.eq('success').astype(int)
df['session_cond'] = df.session.astype(str)

cols = [
    'participant',
    'session',
    'success',
    'time_cond',
    'session_cond',
    'trial',
    'wind',
    'init_dist',
]
df = df.filter(items=cols).sort_values(by=['session_cond','time_cond'])

#%%
rc = {'axes.labelsize': 12, 'axes.titleweight': 'bold'}

with plt.rc_context(rc):
    plt.figure(figsize=[4, 3])
    sns.pointplot(x='session_cond', y='success', hue='time_cond', data=df)
    plt.xlabel('Session number')
    plt.ylabel('Success rate')
    plt.legend(title='Session duration')
    plt.savefig('../figs/success_rates.pdf')

disp(df.groupby(['session_cond', 'time_cond'])[['success']].mean().reset_index())

log_mod = logit("success ~ time_cond * C(session_cond, Treatment(reference='2'))", data=df)
res = log_mod.fit()

# Exponentiated coefficients give us the odds ratio 
# (how odds increase/decrease per unit change in predictor)
oddr = np.exp(res.params)
disp(oddr)
ci_oddr = np.exp(res.conf_int())
disp(ci_oddr)
# Converted to probabilities
probs = oddr / (1 + oddr)

# %%
df = df.sort_values(by=['participant','session','trial'])
df['cum_trial'] = df.groupby('participant').cumcount() + 1
df['wind_abs'] = df.wind.abs()
df['z_cum_trial'] = zscore(df.cum_trial)
df['z_wind_abs'] = zscore(df.wind_abs)
df = df.assign(z_init_dist = zscore(df.init_dist))
filt = df.groupby('participant')[['success']].transform(any).to_numpy()
df_ = df.loc[filt, :]
print(f'Number of participants: {len(df_.participant.unique())}/{len(df.participant.unique())}')

log_mod2 = logit('success ~ z_cum_trial + z_init_dist + z_wind_abs', data=df_)
res2 = log_mod2.fit()
disp(res2.summary())
oddr = np.exp(res2.params)
disp(oddr)
ci_oddr = np.exp(res2.conf_int())
disp(ci_oddr)
probs = oddr / (1 + oddr)

rc = {'axes.labelsize': 12, 'axes.titleweight': 'bold', 'axes.titlesize': 15}

with plt.rc_context(rc):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=[8.25, 3])
    nbins = 10

    plt.sca(ax[0])
    intervals = pd.qcut(df.wind_abs, nbins)
    df['wind_abs_q'] = intervals
    ubs = sorted([f'{round(i.right, 2):.2f}' for i in intervals.unique()])
    # qdf = df.groupby('wind_abs_q')[['success']].mean().reset_index()
    plt.subplot(131)
    sns.pointplot(x='wind_abs_q', y='success', data=df, ci=None)
    plt.xlabel('Absolute wind speed')
    plt.ylabel('Success rate')
    plt.grid(True)
    plt.tick_params(axis='x', bottom=True, length=3, labelrotation=45)
    plt.xticks(np.arange(nbins), ubs, ha='center')
    plt.title('A', loc='center')

    plt.sca(ax[1])
    intervals = pd.qcut(df.init_dist, nbins)
    df['init_dist_q'] = intervals
    ubs = sorted([f'{round(i.right, 1):.1f}' for i in intervals.unique()])
    # qdf = df.groupby('wind_abs_q')[['success']].mean().reset_index()
    sns.pointplot(x='init_dist_q', y='success', data=df, ci=None)
    plt.xlabel('Initialization distance')
    plt.ylabel('')
    plt.grid(True)
    plt.tick_params(axis='x', bottom=True, length=3, labelrotation=45)
    plt.xticks(np.arange(nbins), ubs, ha='center')
    plt.title('B', loc='center')

    plt.sca(ax[2])
    intervals = pd.qcut(df.cum_trial, nbins)
    df['cum_trial_q'] = intervals
    ubs = sorted([int(i.right) for i in intervals.unique()])
    # qdf = df.groupby('wind_abs_q')[['success']].mean().reset_index()
    sns.pointplot(x='cum_trial_q', y='success', data=df, ci=None)
    plt.xlabel('Trial number')
    plt.ylabel('')
    plt.grid(True)
    plt.tick_params(axis='x', bottom=True, length=3, labelrotation=45)
    plt.xticks(np.arange(nbins), ubs, ha='center')
    plt.title('C', loc='center')

    plt.tight_layout()
    plt.savefig('../figs/init_param_effects.pdf')
# %%
