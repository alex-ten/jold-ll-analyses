# %% IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import statsmodels.formula.api as smf

class args():
    ans_data = '../data/clean/answers/clean_answers.csv'
    sess_data = '../data/clean/trials/summarized_pilot_sessions.csv'

index_vars = ['participant', 'session', 'forced']
column_vars = ['presses_mean']

trials_df = pd.read_csv(args.sess_data, index_col=None)
trials_df = trials_df.filter(items = index_vars + column_vars)
trials_df = trials_df.set_index(['participant', 'session']).sort_index()
trials_df = trials_df.loc[trials_df.forced, :].drop(columns=['forced'])
trials_df = trials_df.loc[(slice(None), [1,2]), :]

answers_df = pd.read_csv(args.ans_data, index_col=None)
tlx = answers_df.loc[answers_df.inst.eq('tlx')].set_index(['participant', 'session']).sort_index()
tlx = tlx.filter(items=['qid', 'value']).reset_index()
tlx = tlx.replace({'qid': {1: 'Effort', 2: 'Frustration', 3: 'Mental', 4: 'Performance', 5: 'Physical', 6: 'Temporal'}})
tlx = tlx.pivot(index=['participant', 'session'], columns='qid', values='value').reset_index()

df = trials_df.merge(tlx, on=['participant', 'session']).reset_index()
func = lambda x: np.log(x)
df = df.assign(
    presses_mean = np.log(df.presses_mean),
    # Effort = np.sqrt(df.Effort),
    # Frustration = func(df.Frustration),
    # Mental = func(df.Mental),
    # Performance = func(df.Performance),
    # Physical = func(df.Physical),
    # Temporal = func(df.Temporal)
)

df = df.rename(columns={'presses_mean': 'Mean presses'})
sns.pairplot(
    data = df.drop(columns=['index', 'session']),
    kind = 'reg', 
    plot_kws = {
        'line_kws': {
            'color': 'red', 
            'alpha': .5
        }
    }
)
# %%
