# %%
import pandas as pd
import argparse

# class args():
    # in_path = '../data/raw/answers/pilot.csv'

def main():
    # Load data
    df = pd.read_csv(args.in_path, index_col = None)

    df = df.drop(columns=['id','study']) # Remove primary key column

    # Recode session labels
    replacement_codes = {
        'jold_ll.1.1': '1',
        'jold_ll.2.2': '2',
        'jold_ll.3.3': '3',
    }
    df = df.replace({'session': replacement_codes})

    # Recode jold questions so that they are not mistreated later
    replacement_codes = dict(zip([f'jold-{i}' for i in range(9)], [f'jold-JOLD-{i}' for i in range(9)]))
    df = df.replace({'question': replacement_codes})

    # Remove study prefixes from participant identifiers and question handles
    df = df.assign(
        participant = df.participant.str.strip('jold_ll -- '),
        question = df.question.str.strip('jold-').str.lower()
    )

    # Transform answer values to numeric type
    df = df.assign(value = df.value.astype(int))

    # Split question column into instrument and question id
    df[['inst','qid']] = df.question.apply(lambda col: pd.Series(col.split('-')))
    # df = df.assign(qid = df.qid.astype(int))
    df = df.drop(columns=['question'])

    print(f'Saving data to {args.out_path}')
    df.to_csv(
        path_or_buf = args.out_path,
        sep = ',',
        index = False
    )


parser = argparse.ArgumentParser()
parser.add_argument('in_path', help='relative path to input data')
parser.add_argument('out_path', help='where to save the cleaned data (relative to file)')

args = parser.parse_args()

if __name__=='__main__': main()