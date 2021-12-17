# %% IMPORTS
import pandas as pd

class args():
    in_path = '../data/questions/jold_questions.csv'
    out_path = '../data/questions/clean_questions.csv'


#region: LOAD DATA
# %%
df = pd.read_csv(args.in_path, index_col=None)
df = df.filter(items=['component', 'handle', 'prompt_en', 'annotations_en', 'reverse', 'max_val'])
df = df[df.handle.str.contains('jold') | df.handle.str.contains('sims')]
df = df[~df.handle.str.contains('-dirs')]

# Recode jold questions so that they are not mistreated later
replacement_codes = dict(zip([f'jold-{i}' for i in range(9)], [f'jold-JOLD-{i}' for i in range(9)]))
df = df.replace({'handle': replacement_codes})

# Remove study prefixes from participant identifiers and question handles
df = df.assign(
    question = df.handle.str.strip('jold-').str.lower()
)

# Split question handles into two
df[['inst','qid']] = df.question.apply(lambda col: pd.Series(col.split('-')))

# Rename columns
df = df.rename(columns={'prompt_en': 'prompt', 'annotations_en': 'response'})
df = df.drop(columns=['handle'])

# Save data
print(f'Saving data to {args.out_path}')
df.to_csv(
    path_or_buf = args.out_path,
    sep = ',',
    index = False
)

display(df.head())
#endregion

# %%
