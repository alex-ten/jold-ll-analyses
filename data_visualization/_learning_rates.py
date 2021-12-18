#%% 
import pandas as pd
from IPython.display import display as disp
import numpy as np
from matplotlib import style; style.use('seaborn-whitegrid')
from statsmodels.formula.api import logit
import seaborn as sns

#%%
df1 = pd.read_csv('../data/clean/trials/clean_pilot_trials.csv')
df2 = pd.read_csv('../data/prof_data.csv')

df = df1.merge(df2, left_on='participant', right_on='user', how='right')
df['time_cond'] = df.tenmin.astype(str).replace({'True': 'min10', 'False': 'min20'})
df['success'] = df.outcome.eq('success').astype(int)
df['session_cond'] = df.session.astype(str)

df = df.filter(items=['success', 'time_cond', 'session_cond']).sort_values(by=['session_cond','time_cond'])
sns.pointplot(x='session_cond', y='success', hue='time_cond', data=df)
disp(df.groupby(['session_cond', 'time_cond'])[['success']].mean().reset_index())


log_mod = logit("success ~ time_cond * C(session_cond, Treatment(reference='2'))", data=df)
res = log_mod.fit()

# Exponentiated coefficients give us the odds ratio 
# (how odds increase/decrease per unit change in predictor)
oddr = np.exp(res.params)

# Converted to probabilities
probs = oddr / (1 + oddr)
# %%
