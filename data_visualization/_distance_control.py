#%% 
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as disp
import numpy as np
from matplotlib import style; style.use('seaborn-whitegrid')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.stats as st


def ci95(a):
    dfs = np.sum(~np.isnan(a), axis=0) - 1
    means = np.nanmean(a, axis=0)
    sems = st.sem(a, axis=0, nan_policy='omit')
    zl = zip(dfs, means, sems)
    return np.stack([st.t.interval(alpha=0.95, df=d, loc=m, scale=s) for d, m, s in zl]).T


def aligned_distance_trajectories(df, n, start=True):
    arr = np.full([df.shape[0], n], np.nan)
    return_dict = {'x': arr.copy(), 'y': arr.copy()}
    for i, (_, r) in enumerate(df.iterrows()):
        for d in 'xy':
            dist = np.abs(np.array(r[f'{d}_trail'].split(',')).astype(int) - r[f'plat_{d}'])
            dist = dist[:n] if start else dist[-n:]
            return_dict[d][i, :dist.size] = dist
    return return_dict


def aligned_speed_trajectories(df, n, start=True):
    arr = np.full([df.shape[0], n], np.nan)
    return_dict = {'x': arr.copy(), 'y': arr.copy()}
    for i, (_, r) in enumerate(df.iterrows()):
        for d in 'xy':
            speed = np.array(r[f'{d}_displ'].split(',')).astype(float)
            speed = speed[:n] if start else speed[-n:]
            return_dict[d][i, :speed.size] = speed
    return return_dict


def aligned_angle_trajectories(df, n, start=True):
    z = 0.000001
    return_dict = {'x': np.full([df.shape[0], n], np.nan)}
    for i, (_, r) in enumerate(df.iterrows()):
        dist_x = np.abs(np.array(r['x_trail'].split(',')).astype(int) - r[f'plat_x'])
        dist_y = np.abs(np.array(r['y_trail'].split(',')).astype(int) - r[f'plat_y'])
        angle = np.degrees(np.arctan((dist_x+z)/(dist_y+z)))
        angle[dist_x==0] = 0
        angle[dist_y==0] = 90
        angle = angle[:n] if start else angle[-n:]
        return_dict['x'][i, :angle.size] = angle
    return return_dict


def normalized_distance_trajectories(df, n):
    return_dict = {'x': [], 'y': []}
    for _, r in df.iterrows():
        for d in 'xy':
            dist = np.abs(np.array(r[f'{d}_trail'].split(',')).astype(int) - r[f'plat_{d}'])
            if dist.size >= n:
                return_dict[d].append(
                    np.array([np.mean(_) for _ in np.array_split(dist, n)])
                )
            else:
                continue
    for k in return_dict.keys():
        return_dict[k] = np.stack(return_dict[k], axis=0)
    return return_dict


def normalized_speed_trajectories(df, n):
    return_dict = {'x': [], 'y': []}
    for _, r in df.iterrows():
        for d in 'xy':
            speed = np.array(r[f'{d}_displ'].split(',')).astype(int)
            if speed.size >= n:
                return_dict[d].append(
                    np.array([np.mean(_) for _ in np.array_split(speed, n)])
                )
            else:
                continue
    for k in return_dict.keys():
        return_dict[k] = np.stack(return_dict[k], axis=0)
    return return_dict


def normalized_tilt_trajectories(df, n):
    return_dict = {'x': []}
    for _, r in df.iterrows():
        z = + 0.0000001
        x_displ = np.array(r['x_displ'].split(',')).astype(int)
        y_displ = np.array(r['y_displ'].split(',')).astype(int)
        tilt = np.degrees(np.arctan(y_displ+z/x_displ+z))
        tilt[np.logical_and(x_displ == 0, y_displ > 0)] = 0
        tilt[np.logical_and(x_displ == 0, y_displ == 0)] = 0
        if tilt.size >= n:
            return_dict['x'].append(
                np.array([np.mean(_) for _ in np.array_split(tilt, n)])
            )
        else:
            continue
    for k in return_dict.keys():
        return_dict[k] = np.stack(return_dict[k], axis=0)
    return return_dict 


def line(label, **kwargs):
    return Line2D([0], [0], label=label, **kwargs)


def patch(label, **kwargs):
    return Patch(label=label, **kwargs)


#%% DATA
in_path = '../data/clean/trials/clean_pilot_trials.csv'
df = pd.read_csv(in_path)

# %% Fragmented episode
lstl = {0: '-', 1: ':'}
n = 100
if 'Fragmented episode':
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=[8, 7])
    colors = {
        'Offscreen':'gray',
        'Crash': 'darkred',
        'Success': 'green'
    }
    x = np.arange(1, n+1)
    varname2func_dict = {
        'Distance to platform': normalized_distance_trajectories,
        # 'Angle to platform': normalized_angle_trajectories,
        'Speed': normalized_speed_trajectories,
    }
    cols = ('y_trail','x_trail','plat_y','plat_x','x_displ','y_displ')
    for i1, (var_name, func) in enumerate(varname2func_dict.items()):
        ymax = 0
        for i2, distance in enumerate([1, 2]):
            for i3, outcome in enumerate(['Offscreen','Crash', 'Success']):
                filt = np.abs(df.init_site - df.plat_site) == distance
                filt = filt & df.outcome.eq(outcome.lower())
                filtered = df.loc[filt, cols]
                ndt = func(filtered, n=n)
                for i4, d in enumerate('xy'):
                    if d in ndt.keys():
                        m = np.mean(ndt[d], axis=0)
                        ci = ci95(ndt[d])
                        ax[i1, i2].plot(x, m, c=colors[outcome], ls=lstl[i4])
                        ax[i1, i2].fill_between(x, ci[0], ci[1], alpha=.1, color=colors[outcome])
                        # ax[i4, i2].set_ylim(0, 50)
                        ax[i1, i2].set_xlim(x.min(), x.max())
                        ymax = max(ci.max(), ymax)
        for ax_i in ax[i1, :]: ax_i.set_ylim(0, ymax)
        ax[i1, 0].set_ylabel(var_name)
        
    # Legends and annotations
    ax[0, 0].set_title('Start close')
    ax[0, 1].set_title('Start far')
    ax[-1, 0].set_xlabel('Episode fragment')
    ax[-1, 1].set_xlabel('Episode fragment')
    outcome_lines = [line(k, c=v) for k, v in colors.items()]
    direction_lines = [line('Horizontal', c='k', ls='-'), line('Vertical', c='k', ls=':')]
    ax[0, 0].legend(handles = outcome_lines + direction_lines)

    # Report number of episodes with less than `n` frames
    less_than_n = (df.x_trail.str.count(',') < n).sum()
    total = df.shape[0]
    pless = less_than_n/total
    print(f'{less_than_n}/{total} = {pless}% of trials are less than {n} frames')

    plt.savefig(f'../figs/dist_speed_norm_{n}.pdf')

#%% Start/end of episode
align_to = 'start' # 'start' or 'end'
n = 2000
lstl = {0: '-', 1: ':'}
if f'{align_to} of the episode':
    colors = {
        # 'Offscreen':'gray',
        'Crash': 'darkred',
        'Success': 'green'
    }
    
    x = np.arange(1, n+1)
    v2f = {
        'Distance to platform': aligned_distance_trajectories,
        'Angle to platform': aligned_angle_trajectories ,
        'Speed': aligned_speed_trajectories,
    }
    cols = ('y_trail','x_trail','plat_y','plat_x','x_displ','y_displ')
    
    fig, ax = plt.subplots(nrows=len(v2f.keys()), ncols=2, sharex=True, sharey=False, figsize=[8, 3.5*len(v2f.keys())])
    for i1, (var_name, func) in enumerate(v2f.items()):
        ymax = 0
        for i2, distance in enumerate([1, 2]):
            for i3, outcome in enumerate(colors.keys()):
                filt = np.abs(df.init_site - df.plat_site) == distance
                filt = filt & df.outcome.eq(outcome.lower())
                filtered = df.loc[filt, cols]
                ndt = func(filtered, n=n, start=align_to=='start')
                for i4, d in enumerate('xy'):
                    if d in ndt.keys():
                        m = np.nanmean(ndt[d], axis=0)
                        ci = ci95(ndt[d])
                        ax[i1, i2].plot(x, m, c=colors[outcome], ls=lstl[i4])
                        ax[i1, i2].fill_between(x, ci[0], ci[1], alpha=.1, color=colors[outcome])
                        # ax[i4, i2].set_ylim(0, 50)
                        ax[i1, i2].set_xlim(x.min(), x.max())
                        ymax = max(ci.max(), ymax)
        for ax_i in ax[i1, :]: ax_i.set_ylim(0, ymax)
        ax[i1, 0].set_ylabel(var_name)
        
    # Legends and annotations
    ax[0, 0].set_title('Start close')
    ax[0, 1].set_title('Start far')
    ax[-1, 0].set_xlabel('Frame number')
    ax[-1, 1].set_xlabel('Frame number')
    outcome_lines = [line(k, c=v) for k, v in colors.items()]
    direction_lines = [line('Horizontal', c='k', ls='-'), line('Vertical', c='k', ls=':')]
    ax[0, 0].legend(handles = outcome_lines + direction_lines)

    # Report number of episodes with less than `n` frames
    less_than_n = (df.x_trail.str.count(',') < n).sum()
    total = df.shape[0]
    pless = less_than_n/total
    print(f'{less_than_n}/{total} = {pless}% of trials are less than {n} frames')

    # plt.savefig(f'../figs/dist_speed_{align_to}_{n}.pdf')
#%%