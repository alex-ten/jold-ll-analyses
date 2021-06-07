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
    s_x = np.array(row.x_trail.split(',')).astype(int) - row.plat_x
    s_y = np.array(row.y_trail.split(',')).astype(int) - row.plat_y

    # For each frame compute Euclidean distances to platform
    s = np.sqrt(s_x**2 + s_y**2)

    # Flat-average summary of distances
    row['fad'] = np.mean(s)

    # Weighted-average summary of distances
    relative_weights = np.arange(1, s.size + 1)
    row['wad'] = np.sum(s * (relative_weights/np.sum(relative_weights)))

    return row


def main():
    # Load data
    df = pd.read_csv(args.in_path, index_col = None)

    # Summarize trial trajectories
    df = df.apply(summarize_trial, axis=1)
    df = df.drop(columns=['x_trail', 'y_trail', 'plat_x', 'plat_y'])

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
