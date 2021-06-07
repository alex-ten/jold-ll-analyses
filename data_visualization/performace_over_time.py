#%% 
from numpy.lib import kaiser
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as disp
import seaborn as sns
#%%
in_path = '../data/clean/trials/summarized_pilot_trials.csv'

df = pd.read_csv(in_path)
df = df.assign(time_min = df.time_block / 60)
df = df.query('time_min < 10')
# df = df.loc[df.outcome == 'crash', :]
disp(df.head())
# %%

for k in ['fad','presses', 'time_trial']:
    plt.figure(num=k)

    g = sns.FacetGrid(df,  col='session', hue='participant')
    scatter_kws = dict(
        alpha = .5,
        s = 5,
    )
    g.map(sns.regplot, 'time_block', k, scatter_kws=scatter_kws)
# %%
