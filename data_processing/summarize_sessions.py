# %%
import pandas as pd
import numpy as np
import statsmodels.formula.api as sma
from IPython.display import display as disp
import pickle, os
from scipy.stats import zscore

cwd = os.getcwd()

class args():
    in_path = '../data/clean/trials/summarized_pilot_trials.csv'
    out_path = '../data/clean/trials/summarized_pilot_sessions.csv'


def summarize_session(sess):
    summary = {}
    trial = sess.trial
    half = trial.size // 2

    # Slope of WAD scores
    wadSlope = sma.ols('wad ~ trial', data=sess).fit().params['trial']
    summary['wadSlope'] = wadSlope

    # Slope (of log-odds) for success probability
    summary['sr'] = np.mean(sess.success)
    summary['srd'] = np.mean(sess.success[half:]) - np.mean(sess.success[:half])

    # Means of the second half of session
    summary['wadMeanRecent'] = np.mean(sess.wad[half:])
    summary['pressesMeanRecent'] = np.mean(sess.presses[half:])
    summary['successRateRecent'] = np.mean(sess.success[half:])

    summary['wadDiff'] = np.mean(sess.wad[half:]) - np.mean(sess.wad[half:])
    
    summary['freeChoiceAccepted'] = np.all(sess.index.get_level_values(2))
    
    return pd.Series(summary.values(), index=summary.keys())


# Load data
df = pd.read_csv(args.in_path, index_col = None)
df = df.assign(
    success = df.outcome.str.contains('success').astype(int),
    id = df.participant
)

# Summarize session data
df = df.set_index(['participant','session','forced'])
df = df.sort_values(by=['participant','session','forced','trial'])
df = df.groupby(['participant','session','forced']).apply(summarize_session).reset_index()

print(df.head())

# Save dataset
print(f'Saving data to {args.out_path}')
df.to_csv(
    path_or_buf = args.out_path,
    sep = ',',
    index = False
)
# %%
