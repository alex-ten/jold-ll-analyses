#%%
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as disp
import numpy as np
from matplotlib import style; style.use('seaborn-whitegrid')

#%% DATA
in_path = '../data/clean/trials/clean_pilot_trials.csv'

df = pd.read_csv(in_path)
# disp(df.head())

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=[7, 8])
n = 100
x = np.arange(n)/100
for i, distance in enumerate([1, 2]):
    for j, outcome in enumerate(['Offscreen','Crash', 'Success']):
        filt = np.abs(df.init_site - df.plat_site) == distance
        filt = filt & df.outcome.eq(outcome.lower())
        arr_x = []
        arr_y = []
        arr_xy = []
        for ys, xs in zip(df.loc[filt, 'y_displ'], df.loc[filt, 'x_displ']):
            y_disp = np.array(ys.split(',')).astype(int)
            x_disp = np.array(xs.split(',')).astype(int)
            xy_disp = np.sqrt(y_disp**2 + x_disp**2)
            if xy_disp.size >= n:
                x_disp_std = np.array([np.mean(_) for _ in np.array_split(x_disp, n)])
                y_disp_std = np.array([np.mean(_) for _ in np.array_split(y_disp, n)])
                xy_disp_std = np.array([np.mean(_) for _ in np.array_split(xy_disp, n)])
                arr_x.append(x_disp_std)
                arr_y.append(y_disp_std)
                arr_xy.append(xy_disp_std)
            else:
                continue
        for k, arr in enumerate([arr_xy, arr_x, arr_y]):
            np_arr = np.stack(arr, axis=0)
            m = np.mean(np_arr, axis=0)
            se = np.std(np_arr, axis=0)/np_arr.shape[0]
            mx = np.max(np_arr, axis=0)
            mn = np.min(np_arr, axis=0)
            ax[k,i].plot(x, m, label=outcome)
            ax[k,i].fill_between(x, mn, mx, alpha=.1)
            ax[k,i].set_ylim(0, 50)
            ax[k,i].set_xlim(0, 1)

ax[0,0].set_title('Start close')
ax[0,1].set_title('Start far')

ax[2,1].set_xlabel('Frame fractile')
ax[2,0].set_xlabel('Frame fractile')

ax[0,0].set_ylabel('2D displacement')
ax[1,0].set_ylabel('Horizontal displacement')
ax[2,0].set_ylabel('Vertical displacement')
plt.legend()
plt.savefig('../figs/fractileT_tragecotires')
pless = (df.x_trail.str.count(',') < 100).sum()/df.shape[0]
print(f'{pless}% of trials are less than 100 frames')


#%%  FIRST n EPISODES
in_path = '../data/clean/trials/clean_pilot_trials.csv'

df = pd.read_csv(in_path)
# disp(df.head())

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=[7, 8])
n = 1000
x = np.arange(n)
for i, distance in enumerate([1, 2]):
    for j, outcome in enumerate(['Offscreen','Crash', 'Success']):
        filt = np.abs(df.init_site - df.plat_site) == distance
        filt = filt & df.outcome.eq(outcome.lower())
        # max_len = df.loc[filt, 'x_displ'].str.count(',').max()
        arr_x = np.full([filt.sum(), n], np.nan)
        arr_y = arr_x.copy()
        arr_xy = arr_x.copy()
        for l, (ys, xs) in enumerate(zip(df.loc[filt, 'y_displ'], df.loc[filt, 'x_displ'])):
            x_disp = np.array(xs.split(',')).astype(int)[:n]
            y_disp = np.array(ys.split(',')).astype(int)[:n]
            xy_disp = np.sqrt(y_disp**2 + x_disp**2)

            arr_x[l, :len(x_disp)] = x_disp
            arr_y[l, :len(y_disp)] = y_disp
            arr_xy[l, :len(xy_disp)] = xy_disp

        # x = np.arange(max_len + 1) 
        for k, arr in enumerate([arr_xy, arr_x, arr_y]):
            # np_arr = np.stack(arr, axis=0)
            m = np.nanmean(arr, axis=0)
            se = np.nanstd(arr, axis=0)/np_arr.shape[0]
            mx = np.nanmax(arr, axis=0)
            mn = np.nanmin(arr, axis=0)
            ax[k,i].plot(x, m, label=outcome)
            ax[k,i].fill_between(x, m+se, m-se, alpha=.3)
            # ax[k,i].set_ylim(0, 50)
            # ax[k,i].set_xlim(0, max_len+1)

ax[0,0].set_title('Start close')
ax[0,1].set_title('Start far')

ax[2,1].set_xlabel('Frame')
ax[2,0].set_xlabel('Frame')

ax[0,0].set_ylabel('2D displacement')
ax[1,0].set_ylabel('Horizontal displacement')
ax[2,0].set_ylabel('Vertical displacement')
plt.legend()

pless = (df.x_trail.str.count(',') < 100).sum()/df.shape[0]
print(f'{pless}% of trials are less than 100 frames')
# %%


#%% LAST N EPISODES
in_path = '../data/clean/trials/clean_pilot_trials.csv'

df = pd.read_csv(in_path)
# disp(df.head())

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=False, figsize=[7, 8])
n = 1000
x = np.arange(n)
for i, distance in enumerate([1, 2]):
    for j, outcome in enumerate(['Offscreen','Crash', 'Success']):
        filt = np.abs(df.init_site - df.plat_site) == distance
        filt = filt & df.outcome.eq(outcome.lower())
        # max_len = df.loc[filt, 'x_displ'].str.count(',').max()
        arr_x = np.full([filt.sum(), n], np.nan)
        arr_y = arr_x.copy()
        arr_xy = arr_x.copy()
        for l, (ys, xs) in enumerate(zip(df.loc[filt, 'y_displ'], df.loc[filt, 'x_displ'])):
            x_disp = np.array(xs.split(',')).astype(int)[-n:]
            if len(x_disp) >= n:
                x_disp = np.array(xs.split(',')).astype(int)[-n:]
                y_disp = np.array(ys.split(',')).astype(int)[-n:]
                xy_disp = np.sqrt(y_disp**2 + x_disp**2)
                arr_x[l, -len(x_disp):] = x_disp
                arr_y[l, -len(y_disp):] = y_disp
                arr_xy[l, -len(xy_disp):] = xy_disp

        # x = np.arange(max_len + 1) 
        for k, arr in enumerate([arr_xy, arr_x, arr_y]):
            # np_arr = np.stack(arr, axis=0)
            m = np.nanmean(arr, axis=0)
            se = np.nanstd(arr, axis=0)/np_arr.shape[0]
            mx = np.nanmax(arr, axis=0)
            mn = np.nanmin(arr, axis=0)
            ax[k,i].plot(x, m, label=outcome)
            ax[k,i].fill_between(x, m+se, m-se, alpha=.3)
            # ax[k,i].set_ylim(0, 50)
            # ax[k,i].set_xlim(0, max_len+1)

ax[0,0].set_title('Start close')
ax[0,1].set_title('Start far')

ax[2,1].set_xlabel('Frame')
ax[2,0].set_xlabel('Frame')

ax[0,0].set_ylabel('2D displacement')
ax[1,0].set_ylabel('Horizontal displacement')
ax[2,0].set_ylabel('Vertical displacement')
plt.legend()

pless = (df.x_trail.str.count(',') < 100).sum()/df.shape[0]
print(f'{pless}% of trials are less than 100 frames')

# %%