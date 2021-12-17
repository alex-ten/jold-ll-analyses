#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display as disp
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, zscore
from scipy import stats

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

class args():
    sess_data = '../data/clean/trials/summarized_pilot_sessions2.csv'

df = pd.read_csv(args.sess_data, index_col=None)
df = df.loc[df.forced==True].drop(columns=['forced']).set_index(['participant', 'session'])
df = df[(np.abs(zscore(df)) < 3).all(axis=1)].reset_index()
items = list(df.columns)[2:10]

# df = df.groupby('session').agg(['mean', 'std'], nan_policy='omit').set_index('session')


fig, axes = plt.subplots(3, 3, figsize=[8, 8])
axes = np.reshape(axes, [axes.size, -1]).squeeze()

for n, i in enumerate(items):
    mini = df.loc[:, ('session', i)]
    sns.boxplot(y=i, x='session', data=mini, orient='v', ax=axes[n], whis=[5, 95])

axes[-1].set_visible(False)
plt.tight_layout()
#     ax=axes[0, 0])
# disp(df.head())
# # %%
# # wadSlope	succOddsSlope	pressesSlope	wadMeanRecent	pressesMeanRecent	successRateRecent	
# # surpBeta0	surpWadMu	surpWadSigma	pred
# dv = 'wadMeanRecent'
# formula = f'pred ~ {dv}'
# mdf = df.loc[df.pred.ge(-200)]
# # mdf = df.loc[df.pressesMeanRecent.le(df.pressesMeanRecent.mean()+df.pressesMeanRecent.std()*2)]
# # mdf = df.loc[df.surpBeta0.gt(0)]

# mixed_model = smf.mixedlm(formula=formula, data=mdf, groups=mdf['participant']).fit()
# disp(mixed_model.summary())
# # sns.histplot(x='pred', stat='probability', element='step', data=df.loc[df.session.eq(2)], kde=True)
# sns.lmplot(x=dv, y='pred', data=mdf, palette='tab10')

# %%
