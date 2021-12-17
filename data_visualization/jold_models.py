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


def pval_stars(p):
    if p < 0.1:
        return '.'
        if p <= 0.05:
            return'*'
            if p <= 0.01:
                return '**'
    else:
        return 'n.s.'

#region: DATA
# %%

varz = [
    'wad_slope',
    'sodds_slope',
    'presses_slope',
    'wad_mean2',
    'presses_mean2',
    'sr2',
    'surp_1_999',
    'surp_1_1'
]
select = ['participant', 'session', 'forced'] + varz
tri = pd.read_csv(args.sess_data, index_col=None)
tri = tri.filter(items=select)
tri = tri.set_index(['participant', 'session']).sort_index()
tri = tri.loc[tri.forced, :].drop(columns=['forced'])
tri = tri.loc[(slice(None), [1,2]), :]

ans = pd.read_csv(args.ans_data, index_col=None)
#endregion

#region: JOLD1 CORRELATIONS
# %%
jold1 = ans.loc[ans.inst.eq('jold') & ans.qid.eq(1)]
jold1 = jold1.set_index(['participant', 'session']).sort_index().loc[:, 'value'] - 6
df = tri.merge(jold1, on=['participant', 'session']).reset_index()
# df = df.loc[df.value > 0, :]
# df = df.groupby(['participant']).filter(lambda x: len(x) > 1)

for k in varz:
    display(smf.ols(f'value ~ {k}', data=df.loc[df.session==1, :]).fit().summary())
    mm = smf.mixedlm(f'value ~ {k}', df, groups=df['session'], re_formula=f'~{k}')
    mdf = mm.fit(method=['lbfgs'])
    display(
        pd.DataFrame(
            {
                'Param' : round(mdf.params, 3),
                'p-value': round(mdf.pvalues, 3),
                'Sign.': [pval_stars(p) for p in mdf.pvalues]
            }
        )
    )

# pair_df = df.set_index('session').filter(items=varz+['value'])
# sns.pairplot(pair_df.loc[1, :], kind='reg')
# plt.gcf().savefig('../figs/jold1_sess1_pairplot.png')

pair_df = df.set_index('session').filter(items=varz+['value'])
sns.pairplot(pair_df.loc[1, :])
# plt.gcf().savefig('../figs/jold1_sess2_pairplot.png')
#endregion

#region: JOLD2 HISTOGRAMS
# %%
jolds = ans.loc[ans.inst.eq('jold')].set_index(['participant', 'session']).sort_index()
jolds = jolds.reset_index().pivot(index=['participant','inst','qid'], columns='session', values='value') - 6

_ = slice(None)
sns.regplot(x=jolds.loc[(_, _, 1), 2].dropna(), y=jolds.loc[(_, _, 2), 2])
plt.xlabel('JOLD 1')
plt.ylabel('JOLD 2')
plt.gca().axes.set_aspect('equal')
#endregion
# %%



#region: JOLD1 CORRELATIONS
# %%
tlx = ans.loc[ans.inst.eq('tlx')].set_index(['participant', 'session']).sort_index()
tlx = tlx.filter(items=['qid', 'value']).reset_index()
tlx = tlx.pivot(index=['participant', 'session'], columns='qid', values='value').reset_index()
df = tri.merge(tlx, on=['participant', 'session']).reset_index()

pair_df = df.set_index('session')
sns.pairplot(pair_df.loc[1, :], kind='reg', plot_kws={'line_kws': {'color': 'red', 'alpha': .5}})
#endregion
# %%
