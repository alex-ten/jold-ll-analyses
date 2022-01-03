# %% IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style; style.use('seaborn-whitegrid')
from matplotlib.ticker import MaxNLocator
from IPython.display import display
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

# %% JOLD retro and prosp distributions and correlation
ans_df = pd.read_csv('../data/clean/answers/clean_answers.csv', index_col=None)
jolds = ans_df.loc[ans_df.inst.eq('jold')].set_index(['participant', 'session']).sort_index()
jolds = jolds.loc[jolds.qid.eq(1) | jolds.qid.eq(8), :]
jolds = jolds.reset_index().drop(columns=['inst'])
jolds = jolds.pivot(index=['participant', 'session'], columns='qid', values='value') - 6
jolds = jolds.reset_index()

nas = jolds.loc[:, 8].isna().sum()
N = jolds.shape[0]
print(f'{nas}/{N} rows ({nas/N*100}%) are missing jold-8 ratings')
jolds = jolds.loc[~jolds.loc[:, 8].isna(), :]
bins = np.arange(-5, 6)

rc = {'axes.labelsize': 15, 'axes.titleweight': 'bold'}
with plt.rc_context(rc):
    g = sns.JointGrid(data=jolds, x=1, y=8, xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), height=5)
    g.plot_joint(sns.regplot, scatter_kws=dict(s=150, alpha=.2))
    g.plot_marginals(sns.histplot, bins=bins, stat='percent')
    g.refline(x=0, y=0)
    g.set_axis_labels(xlabel='Retrospective judgment rating', ylabel='Prospective judgment rating')
    g.ax_marg_y.tick_params(labeltop=True)
    g.ax_marg_y.grid(True, axis='x')
    g.ax_marg_y.set_xlabel('%')
    g.ax_marg_y.xaxis.set_major_locator(MaxNLocator(4))
    g.ax_marg_y.set_visible(True)

    g.ax_marg_x.tick_params(labelleft=True)
    g.ax_marg_x.grid(True, axis='y')
    g.ax_marg_x.set_ylabel('%')
    g.ax_marg_x.yaxis.set_major_locator(MaxNLocator(4))
    
r, pval = spearmanr(jolds.loc[:, 1], jolds.loc[:, 8])
dof = jolds.loc[:, 1].shape[0] - 2
print(f'Pearson\s r({dof}) = {round(r, 3):.3f}, p = {round(pval, 3):.3f}')
plt.savefig('../figs/jolds_corr.pdf')
# %%

