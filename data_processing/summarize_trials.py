import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('in_path', help='relative path to input data')
parser.add_argument('out_path', help='where to save the cleaned data (relative to file)')

args = parser.parse_args()


def summarize_trial(row):
    '''[summary]

    Parameters
    ----------
    row : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''

    # For each frame compute signed 1-d distances from platform
    s_x = np.array(row.x_trail.split(',')).astype(int)
    s_y = np.array(row.y_trail.split(',')).astype(int)

    d_x = s_x - row.plat_x
    d_y = s_y - row.plat_y

    # For each frame compute Euclidean distances to platform
    d = np.sqrt(d_x**2 + d_y**2)

    # Flat-average summary of distances
    row['fad'] = np.mean(d)

    # Weighted-average summary of distances
    relative_weights = np.arange(1, d.size + 1)
    row['wad'] = np.sum(d * (relative_weights/np.sum(relative_weights)))

    # Speed
    v_x = np.abs(s_x[:1] - s_x[1:])
    v_y = np.abs(s_y[:1] - s_y[1:])

    row['ahs'] = np.mean(v_x)
    row['avs'] = np.mean(v_y)

    return row


def main():
    # Load data
    df = pd.read_csv(args.in_path, index_col = None)

    # Summarize trial trajectories
    df = df.apply(summarize_trial, axis=1)
    df = df.drop(columns=['x_trail', 'y_trail', 'plat_x', 'plat_y'])
    sign = 2 * df.outcome.str.contains('success').astype(int) - 1
    df = df.assign(
        cwad = df.wad * sign,
        cfad = df.fad * sign
    )

    print(df.head())
    # Save dataset
    print(f'Saving data to {args.out_path}')
    df.to_csv(
        path_or_buf = args.out_path,
        sep = ',',
        index = False
    )


if __name__=='__main__': 
    print('\nSummarizing clean trial data...')
    main()
