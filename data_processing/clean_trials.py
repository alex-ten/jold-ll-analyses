import argparse
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('in_path', help='relative path to input data')
parser.add_argument('out_path', help='where to save the cleaned data (relative to file)')

args = parser.parse_args()


def split_xy(val):
    '''When applied to pd.Series that consists of 2 comma-separated string numbers, this
    function splits each value of the series into two ints

    Parameters
    ----------
    val : str
        A string of two comma separated integer numbers (e.g., "111,222")

    Returns
    -------
    list 
        Returns a list of 2 integers (e.g., [111, 222])
    '''
    return pd.Series(val.split(',')).astype(int)


def pop_1st_v(val):
    '''When applied to pd.Series that consists of strings of comma-separated numbers, this
    function does 2 things: 
        1) Applies ''.split(',') to each string 
        2) Converts the resulting list to an int numpy array to remove the first 10 elements
        3) Joins values back into a string

    Parameters
    ----------
    val : str
        string of comma-separated integers

    Returns
    -------
    str
        input string, less the first value
    '''
    return ','.join(val.split(',')[1:])


def displacement_1d(x):
    xnum = np.array(x.split(',')).astype(int)
    displ = np.abs(xnum[:-10] - xnum[10:])
    return ','.join(displ.astype(str).tolist())


def displacement_2d(df):
    xnum = np.array(df.x_displ.split(',')).astype(int)
    ynum = np.array(df.y_displ.split(',')).astype(int)
    displ2d = np.sqrt(xnum**2 + ynum**2)
    return ','.join(displ2d.astype(str).tolist())


def main():
    # Load raw dataset
    df = pd.read_csv(args.in_path, parse_dates=['date'])
    df = df.drop(columns=['id']) # Remove primary key column

    # Remove 'termination' trials and buggy 'unknown' trials
    df = df.loc[~df.outcome.eq('termination'), :]
    df = df.loc[~df.outcome.eq('unknown'), :]

    # Recode session labels
    replacement_codes = {
        'jold_ll.1.1': '1',
        'jold_ll.2.2': '2',
        'jold_ll.3.3': '3',
    }
    df = df.replace({'session': replacement_codes})

    # Round up timestamps to nearest second
    df = df.assign(date = df.date.dt.round('1s').dt.strftime("%m/%d %H:%M:%S"))

    # Remove study prefixes from participant identifiers
    df = df.assign(participant = df.participant.str.strip('jold_ll -- '))

    # Split plat_xy into two separate columns
    df[['plat_x','plat_y']] = df.plat_xy.apply(split_xy)
    df = df.drop(columns=['plat_xy'])

    # Check for nans, report corrupted trials, drop nans
    mask = df.isna().any(axis=1) | ~df.x_trail.str.contains(',', na=False) | ~df.y_trail.str.contains(',', na=False)
    bad_rows = df[mask]
    if not bad_rows.empty:
        pids = bad_rows.participant.unique()
        print(f' {pids.size} participant(s) had NaN rows:')
        for pid in pids:
            print(f'    {pid}: {bad_rows.participant.eq(pid).sum()} row(s)')
        print(f' Total of {mask.sum()} deleted.')
    df = df.loc[~mask]

    # Convert strings of comma-separated x and y coordinates into numpy int arrays
    # and cut the first 10 frames of each array (i.e., remove coordinates of the first 10 frames)
    df = df.assign(x_trail = df.x_trail.apply(pop_1st_v))
    df = df.assign(y_trail = df.y_trail.apply(pop_1st_v))

    # Calculate 1D displacement
    df = df.assign(
        x_displ = df.x_trail.apply(displacement_1d),
        y_displ = df.y_trail.apply(displacement_1d)
    )
    # # Calculate 2D displacement
    # df = df.assign(
    #     xy_displ = displacement_2d,
    # )

    # Remove participants who finished no more than 5 trials in total
    leavers = df.groupby('participant').filter(lambda x: x.shape[0] <= 5).participant.unique()
    n_left = len(leavers)
    print(f'Removed {n_left} participants with no more than 5 trials in total:')
    print(leavers)
    df = df.groupby('participant').filter(lambda x: x.shape[0] > 5)
    
    # Save dataset
    print(f'Saving data to {args.out_path}')
    df.to_csv(
        path_or_buf = args.out_path,
        sep = ',',
        index = False,
    )


if __name__=='__main__': 
    print('\nCleaning raw data...')
    main()

