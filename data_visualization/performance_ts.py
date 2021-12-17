#region: IMPORTS
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class args():
    in_path = '../data/clean/trials/summarized_pilot_trials.csv'
#endregion

#region: LOAD DATA
# %%
df = pd.read_csv(args.in_path, index_col = None)
df = df.assign(success = df.outcome.str.contains('success').astype(int))
df = df.set_index(['participant','session','forced'])
df = df.sort_values(by=['participant','session','forced','trial'])
df = df.groupby('participant').filter(lambda x: x.shape[0] >= 5)
participants = list(df.index.get_level_values(0).unique())
#endregion

#region: VISUALIZE AUTOCORRELATION
# %%
fig = plt.figure('wad time series', figsize=[18,11])
Nr, Nc = 5, 8
for i, p in enumerate(participants, 1):
    dur = 20 if sdf.time_block.gt(720).any() else 10
    ax = fig.add_subplot(Nr, Nc, i)
    sdf = df.loc[(p, slice(None), True), :]
    if (i-1) % Nc != 0:
        ax.set_ylabel('')
        # ax.tick_params(labelleft=False)
    y, ylim = 'presses', 160
    for t in sdf.groupby('session').tail(1)[['trial']].values.squeeze().cumsum():
        ax.axvline(t, ls = ':', color = 'k')
    sdf = sdf.assign(
        gtrial = np.arange(1, sdf.shape[0]+1),
        sr = sdf.success.rolling(min_periods=5, window=5).mean(),
        wad_smooth = sdf.wad.rolling(min_periods=5, window=5).mean()
    )
    sns.lineplot(x='gtrial', y=y, data=sdf, ax=ax, color='red' if dur == 20 else 'blue')
    # ax.set_ylim(0, ylim)
    textlabel = p if len(p) < 10 else p[:10]+'/'
    textlabel = f'({i-1}) {textlabel}\n#sess={sdf.index.get_level_values(1).max()}, dur={dur}'
    ax.text(.5, .85, textlabel, transform=ax.transAxes, ha='center', va='center')
fig.tight_layout()
# fig.savefig(f'../figs/{y}_timeseries.png')
#endregion
# %%
