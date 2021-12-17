# %% IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant

class args():
    ans_data = '../data/clean/answers/clean_answers.csv'
    sess_data = '../data/clean/trials/summarized_pilot_sessions.csv'
    questions_data = '../data/questions/clean_questions.csv'



#region: DATA
# %%
qs = pd.read_csv(args.questions_data, index_col=None)
ans = pd.read_csv(args.ans_data, index_col=None)

df = pd.merge(ans, qs.filter(items=['inst', 'qid', 'component', 'reverse', 'max_val']), on=['inst', 'qid'])
# df = df.loc[df.session!=3, :]
print(df.inst.unique())
del ans

# CALCULATE MEASURES
mask = sum([df.inst.str.contains(s) for s in ('jold', 'sims', 'tlx', 'mslq')])
df = df[mask.astype(bool)]
df.value.where(~df.reverse, (df.max_val+1)-df.value, inplace=True)
df = df.filter(items=['participant', 'session', 'inst', 'component', 'qid', 'value'])
df = df.sort_values(by=['participant', 'session', 'inst', 'component', 'qid'])
# df = df.loc[~(df.inst.eq('sims') & df.qid.eq(6)), :] # feels good
# df = df.loc[~(df.inst.eq('sims') & df.qid.eq(2)), :] # fun
# df = df.loc[~(df.inst.eq('sims') & df.qid.eq(15)), :] # pleasant
df = df.loc[~(df.inst.eq('sims') & df.qid.eq(11)), :] # interesting
df = df.groupby(['participant', 'session', 'inst', 'component']).mean().reset_index()
#endregion

#region: NASA-TLX
# %%
tlx = df.loc[df.inst.eq('tlx'), :]
display(tlx.groupby(['session', 'component'])[['value']].describe())

plt.close('all')
plt.figure(num='NASA-TLX', figsize=[8, 8])
sns.boxplot(x='value', y='component', hue='session',
                 data=tlx, palette='Set1')
sns.swarmplot(
    x='value', y='component', data=tlx, hue='session', palette='Set1', 
    dodge=True, edgecolor='gray', linewidth=1, alpha=.5
)
plt.xlim(0, 21)
plt.ylabel('NASA-TLX Component')
plt.xlabel('Score')
# plt.gcf().savefig('../figs/tlx_by_session.png')
#endregion

#region: MSLQ
# %%
mslq = df.loc[df.inst.eq('mslq'), :]
trait_scales = ['Cognitive and Metacognitive Strategies: Metacognitive Self-Regulation', 'Resource Management Strategies: Effort Regulation']
mask = np.logical_or(*[mslq.component.eq(i) for i in trait_scales])
traits = mslq.loc[mask, :]
mslq = mslq.loc[~mask, :]
disp(mslq.groupby(['session', 'component'])[['value']].describe().round(3))

plt.close('all')
plt.figure(num='MSLQ', figsize=[8, 8])
sns.boxplot(x='value', y='component', hue='session',data=mslq, palette='Set1')
sns.swarmplot(
    x='value', y='component', data=mslq, hue='session', palette='Set1', 
    dodge=True, edgecolor='gray', linewidth=1, alpha=.5
)# plt.xlim(0, 21)
plt.ylabel('MSLQ Component')
plt.xlabel('Score')
plt.gcf().savefig('../figs/mslq_by_session.png')


def print_component_questions(component, df):
    print(component)
    for i, prompt in enumerate(df.loc[df.component==component, 'prompt'].unique()):
        print(f'\t{i+1}) {prompt}')

for c in mslq.component.unique():
    print_component_questions(c, qs)
#endregion

#region: SIMS
# %%
sims = df.loc[df.inst.eq('sims'), :]
display(sims.groupby(['session', 'component'])[['value']].describe().round(3))

plt.close('all')
plt.figure(num='SIMS', figsize=[8, 5])
sns.boxplot(x='value', y='component', hue='session', data=sims, palette='Set1')
sns.swarmplot(
    x='value', y='component', data=sims, hue='session', palette='Set1', 
    dodge=True, edgecolor='gray', linewidth=1, alpha=.5
)
# plt.xlim(0, 21)
plt.ylabel('SIMS Component')
plt.xlabel('Score')
plt.gcf().savefig('../figs/sims_by_session.png')


def print_component_questions(component, df):
    print(component)
    for i, prompt in enumerate(df.loc[df.component==component, 'prompt'].unique()):
        print(f'\t{i+1}) {prompt}')

for c in sims.component.unique():
    print_component_questions(c, qs)
#endregion

#region: JOLDS AND MOTIVATION
# %%
predictor = df.loc[df.component=='Long-term']
ind = predictor.set_index(['participant', 'session']).index
mslq_sefl = df.loc[df.component=='Self-Efficacy for Learning and Performance'].set_index(['participant', 'session'])
mslq_ego = df.loc[df.component=='Extrinsic Goal Orientation'].set_index(['participant', 'session'])
mslq_clb = df.loc[df.component=='Control of Learning Beliefs'].set_index(['participant', 'session'])
mslq_tv = df.loc[df.component=='Task Value'].set_index(['participant', 'session'])
sims_amot = df.loc[df.component=='Amotivation'].set_index(['participant', 'session'])
sims_extr = df.loc[df.component=='External regulation'].set_index(['participant', 'session'])
sims_idr = df.loc[df.component=='Identified regulation'].set_index(['participant', 'session'])
sims_im = df.loc[df.component=='Intrinsic motivation'].set_index(['participant', 'session'])

d = {
    'predictor': predictor.value.values,
    'mslq_sefl': mslq_sefl.loc[ind, :].value.values,
    'mslq_ego': mslq_ego.loc[ind, :].value.values,
    'mslq_clb': mslq_clb.loc[ind, :].value.values,
    'mslq_tv': mslq_tv.loc[ind, :].value.values,
    'sims_amot': sims_amot.loc[ind, :].value.values,
    'sims_extr': sims_extr.loc[ind, :].value.values,
    'sims_idr': sims_idr.loc[ind, :].value.values,
    'sims_im': sims_im.loc[ind, :].value.values,
}
df_ = pd.DataFrame(d)


# g = sns.pairplot(df_, kind='reg', plot_kws={'line_kws': {'color': 'red'}})

dvs = list(d.keys())

ivs, bs, ts, ps, sigls = [], [], [], [], []
for dv_key in list(d.keys())[1:]:
    m = OLS(endog=df_.loc[:, dv_key], exog=add_constant(df_.predictor)).fit()

    bs.append(round(m.params[1], 3))
    ts.append(round(m.tvalues[1], 3))
    p = m.pvalues[1]
    ps.append(round(p, 3))
    if p < 0.1:
        s = '.'
        if p <= 0.05:
            s = '*'
            if p <= 0.01:
                s = '**'
                
    else:
        s = 'n.s.'
    sigls.append(s)
rdf = pd.DataFrame(
    {
        'DV': dvs[1:],
        'b' : bs,
        't' : ts,
        'p' : ps,
        'Sign.': sigls

    }
)
display(rdf)
print(f'Note: df = ({m.df_model}, {m.df_resid})')

#endregion

# %%

# %%
