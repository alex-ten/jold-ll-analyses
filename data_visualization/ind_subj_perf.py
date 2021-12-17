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
# df = df.query('time_min < 10')
# df = df.loc[df.outcome == 'crash', :]
disp(df.head())
# %%
participants = df.participant.unique()
username = participants[4]
print(f'Username: {username}')
one = df.loc[df.participant.eq(username)]
for k in ['cwad']:
    plt.figure(num=k)

    g = sns.FacetGrid(one, col='session', hue='participant')
    scatter_kws = dict(
        alpha = .5,
        s = 5,
    )
    g.map(sns.regplot, 'trial', k, scatter_kws=scatter_kws)
    
    plt.figure(num=k+'2')
    sns.histplot(one, x=k, hue='session', element='step')
# %%
