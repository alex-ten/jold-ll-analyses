#%%
import argparse
import pandas as pd
import numpy as np
from scipy.stats import zscore, linregress
from IPython.display import display as disp

# parser = argparse.ArgumentParser()
# parser.add_argument('in_path', help='relative path to input data')
# parser.add_argument('out_path', help='where to save the cleaned data (relative to file)')

# args = parser.parse_args()


def summarize_session(sess, rw=5):
    trial = sess.trial - sess.trial.min() + 1
    zfad = zscore(sess.fad)
    zwad = zscore(sess.wad)
    roll = sess[['fad','wad','success']].rolling(window=rw, axis=0)
    roll_mean = roll.mean().dropna()
    roll_var = roll.var().dropna()

    fad_slope = linregress(trial, zfad).slope  # rate of FAD change
    fad_accel = np.polyfit(trial, zfad, 2)[0]  # rate of FAD acceleration
    fad_varr = linregress(trial[rw-1:], zscore(roll_var.fad)).slope  # rate of FAD variance change

    wad_slope = linregress(trial, zwad).slope  # rate of WAD change
    wad_accel = np.polyfit(trial, zwad, 2)[0]  # rate of WAD acceleration
    wad_varr = linregress(trial[rw-1:], zscore(roll_var.wad)).slope  # rate of WAD variance change

    sr_slope = linregress(trial[rw-1:], zscore(roll_mean.success)).slope  # change of success rate

    

    summary = {
        'fad_slope': fad_slope,
        'fad_accel': fad_accel,
        'fad_varr': fad_varr,
        'wad_slope': wad_slope,
        'wad_accel': wad_accel,
        'wad_varr': wad_varr,
        'sr_slope': 0 if np.isnan(sr_slope) else sr_slope,
    }
    return pd.Series(summary.values(), index=summary.keys())
#%%

def main():
    # Load data
    args_in_path = '../data/clean/trials/summarized_pilot_trials.csv'
    df = pd.read_csv(args_in_path, index_col = None)
    df = df.assign(success = df.outcome.str.contains('success').astype(int))

    # Summarize session
    df = df.set_index(['participant','session','forced'])
    df = df.groupby(['participant','session','forced']).apply(summarize_session)

    # # Save dataset
    # print(f'Saving data to {args.out_path}')
    # df.to_csv(
    #     path_or_buf = args.out_path,
    #     sep = ',',
    #     index = False
    # )
    return df

df = main()
disp(df)
# if __name__=='__main__': main()

# %%
