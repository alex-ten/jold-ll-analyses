#%% 
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as disp
import numpy as np
import scipy.stats as st

#%%
in_path = '../data/clean/trials/clean_pilot_trials.csv'

df = pd.read_csv(in_path)
disp(df.head())
# %% Visualize first and last n frames for different outcomes
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

n = 50
x = np.arange(n)
for ax_i, outcome in enumerate(['crash', 'success', 'offscreen']):
    y_displ = df.loc[df.outcome.eq(outcome), 'y_displ']
    x_displ = df.loc[df.outcome.eq(outcome), 'x_displ']
    arr = np.full([len(y_displ), n], np.nan)
    arr2 = np.full([len(y_displ), n], np.nan)
    cc = 0
    for i, (ys, xs) in enumerate(zip(y_displ, x_displ)):
        yvals = np.array(ys.split(',')).astype(int)
        if yvals.size < n:
            cc += 1
            continue
        xvals = np.array(ys.split(',')).astype(int)
        xy_disp = np.sqrt(xvals**2 + yvals**2)
        arr[i, :] = xy_disp[-n:]
        arr2[i, :] = xy_disp[:n]
    print(cc)
    m = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sum(~np.isnan(arr), axis=0)
    ax2.plot(x, m, label=outcome)
    ax2.fill_between(x, m+se, m-se, alpha=.1)
    ax2.set_ylim(0,21)

    m = np.nanmean(arr2, axis=0)
    se = np.nanstd(arr2, axis=0) / np.sum(~np.isnan(arr), axis=0)
    ax1.plot(x, m, label=outcome)
    ax1.fill_between(x, m+se, m-se, alpha=.1)
    ax1.set_ylim(0,21)
plt.legend()
# %% Try splitting each trial into 100 equal parts and calculate mean in each part (0 to 9)
fig = plt.figure(figsize=[8, 4])
ax = [fig.add_subplot(1,2,i+1) for i in range(2)]
n = 100
x = np.arange(n)
for ax_i, outcome in enumerate(['offscreen','crash', 'success']):
    for k, idist in enumerate([0, 1]):
        # filt = np.abs(df.init_site - df.plat_site) == idist
        filt = np.abs(df.wind) > 1.5 if idist else np.abs(df.wind) < 1.5
        y_displ = df.loc[df.outcome.eq(outcome) & filt, 'y_displ']
        x_displ = df.loc[df.outcome.eq(outcome) & filt, 'x_displ']
        arr = []
        cc = 0
        for i, (ys, xs) in enumerate(zip(y_displ, x_displ)):
            yvals = np.array(ys.split(',')).astype(int)
            xvals = np.array(xs.split(',')).astype(int)
            xy_disp = np.sqrt(xvals**2 + yvals**2)
            xy_disp = xvals
            if xy_disp.size > n:
                xy_disp_standard = np.array([np.mean(i) for i in np.array_split(xy_disp, n)])
                arr.append(xy_disp_standard)
            else:
                cc += 1
                continue
        print(cc)
        arr = np.stack(arr, axis=0)
        m = np.mean(arr, axis=0)
        se = np.std(arr, axis=0)/arr.shape[0]
        ax[k].plot(x, m, label=outcome)
        ax[k].fill_between(x, m+se, m-se, alpha=.1)
        ax[k].set_ylim(0,30)
plt.legend()
# %%
