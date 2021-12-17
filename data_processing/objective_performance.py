# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

class args():
    in_path = '../data/clean/trials/summarized_pilot_trials.csv'
    out_path = '../data/clean/trials/summarized_pilot_sessions.csv'

# %%
df = pd.read_csv(args.in_path, index_col = None)
df = df.assign(
    success = df.outcome.str.contains('success').astype(int),
    wind_discrete = np.abs(np.round(df.wind)),
    init_dist_discrete = np.abs(df.init_site - df.plat_site).astype(str)
)

display(df.head())
# %%
fig, ax = plt.subplots(num='success trials')
v = 'end_dist'
h = 'init_dist_discrete'
# sns.stripplot(y='success', x='time_trial', data=df, ax=ax, alpha=.1)
sns.boxplot(x=v, y='success', hue=h, dodge=True, orient='h', whis=[1, 99],  data=df, ax=ax)
# %%
