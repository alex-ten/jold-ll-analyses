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

ans = pd.read_csv(args.ans_data, index_col=None)

# %%
jolds = ans.loc[ans.inst.eq('jold')].set_index(['participant', 'session']).sort_index()
jolds = jolds.reset_index().pivot(index=['participant','inst','qid'], columns='session', values='value') - 6
sns.distplot(
    jolds.loc[(slice(None), slice(None), 1), 1].dropna(),
    bins=np.arange(-7, 7)
)

# _ = slice(None)
# sns.regplot(x=jolds.loc[(_, _, 1), 2].dropna(), y=jolds.loc[(_, _, 2), 2])
# plt.xlabel('JOLD 1')
# plt.ylabel('JOLD 2')
# plt.gca().axes.set_aspect('equal')
# # %%

# %%
