# Correlational analysis between competence judgments (NASA-TLX) and candidate explanatory variables
# First, visualize average competence judgments for each session (this is just to get a feel for the distributions)
# Then, for each session, make a scatterplot competence x metric
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display as disp
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

class args():
    ans_data = '../data/clean/answers/clean_answers.csv'
    sess_data = '../data/clean/trials/summarized_pilot_sessions2.csv'
    questions_data = '../data/questions/clean_questions.csv'


qs = pd.read_csv(args.questions_data, index_col=None)
ans = pd.read_csv(args.ans_data, index_col=None)
ivs = pd.read_csv(args.sess_data, index_col=None)
iv_items = list(ivs.columns)[3:]

df = pd.merge(ans, qs.filter(items=['inst', 'qid', 'component', 'reverse', 'max_val']), on=['inst', 'qid'])
# df = df.loc[df.session!=3, :]
print(df.inst.unique())
del ans

# CALCULATE MEASURES
mask = df.inst.str.contains('mslq')
mask2 = df.inst.str.contains('sims')
df2 = df[mask2.astype(bool)]
df = df[mask.astype(bool)]
df.value.where(~df.reverse, (df.max_val+1)-df.value, inplace=True)
df2.value.where(~df2.reverse, (df2.max_val+1)-df2.value, inplace=True)
df = df.filter(items=['participant', 'session', 'inst', 'component', 'qid', 'value'])
df = df.sort_values(by=['participant', 'session', 'inst', 'component', 'qid'])
df2 = df2.filter(items=['participant', 'session', 'inst', 'component', 'qid', 'value'])
df2 = df2.sort_values(by=['participant', 'session', 'inst', 'component', 'qid'])
# df = df.loc[~(df.inst.eq('sims') & df.qid.eq(11)), :] # interesting
df = df.groupby(['participant', 'session', 'inst', 'component']).mean().reset_index()
df2 = df2.groupby(['participant', 'session', 'inst', 'component']).mean().reset_index()

q_items = list(df.component.unique())
q_items2 = list(df2.component.unique())

# df = df.loc[df.component.eq('Frustration'), :]
df = df.pivot(index=['participant', 'session'], columns='component', values='value')
df = df.merge(ivs, on=['participant','session'])
df = df.loc[df.forced.eq(True)]
# df = df.assign(pred = df.value - 0)

df2 = df2.pivot(index=['participant', 'session'], columns='component', values='value')
df2 = df2.merge(ivs, on=['participant','session'])
df2 = df2.loc[df2.forced.eq(True)]


disp(df.head())

# %% Correlation matrix
# df2 = df.filter(items=['Recent', 'Prospective'])
# df2.loc[:, 'Recent'] = df2.Recent - 6
df1 = df.filter(items=q_items)
df2 = df2.filter(items=q_items2)
corr_df = pd.concat([df1, df2], axis=1).corr().filter(df1.columns).filter(df2.columns, axis=0)
p_vals = calculate_pvalues(pd.concat([df1, df2], axis=1)).filter(df1.columns).filter(df2.columns, axis=0)

plt.matshow(corr_df, cmap='bwr', vmin=-1, vmax=1)
plt.xticks(range(df1.columns.size), df1.columns, fontsize=14, rotation=45, ha='left', va='bottom')
plt.yticks(range(df2.columns.size), df2.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix\n(colors indicate Pearson correlation coefficient)', fontsize=16);# %%
disp(corr_df.round(3))
p_vals_str = p_vals.astype(str)
p_vals_str = p_vals_str.where(p_vals > 0.05, p_vals_str+'*')
disp(p_vals_str.round(5))
# %%