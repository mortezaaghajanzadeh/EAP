#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import statsmodels.api as sm
# %% Loading the Matlab data
data_path = r"Data"
data_out = r"Out"
#%%
data_set  = sio.loadmat(data_path + "/Assignment2Ex3Data.mat")
# %%

map_columns = ['permno','months']

data_columns = ['AnnouncementReturn', 'IdioVolAHT', 'PayoutYield', 'RD', 'VolMkt', 'exchcd', 'me', 'prc', 'ret']


def clean_df(data_set, column):
    df = pd.DataFrame(data_set[column]).melt().rename(columns={'variable':'permno', 'value':column})
    df['permno'] = df['permno'].astype(int) 
    df['t'] = df.groupby('permno').cumcount()
    return df

for i in data_columns:
    if i == data_columns[0]:
        df = clean_df(data_set, i)
    else:
        df = df.merge(clean_df(data_set, i), on=['permno','t'], how='left')
mapping_dict = dict(zip(range(0, len(data_set['permno'].flatten())), data_set['permno'].flatten()))
df['permno'] = df['permno'].map(mapping_dict)
mapping_dict = dict(zip(range(0, len(data_set['months'].flatten())), data_set['months'].flatten()))
df['t'] = df['t'].map(mapping_dict)
df.dropna(subset=['ret','me'], inplace=True)
#%% VolMkt
def get_portfolios(df, sorting_car, number_of_portfolios,weighting = 'Equal'):
    portfoli_df = df.dropna(subset=[sorting_car])[
        ['t', 'permno', sorting_car, 'me', 'ret']
    ].copy()
    portfoli_df['portfolios'] = portfoli_df.groupby('t')[sorting_car].transform(lambda x: pd.qcut(x, number_of_portfolios, labels=False)) 
    portfoli_df['portfolios'] = portfoli_df['portfolios'] + 1   # The highest value is the highest portfolio
    if weighting == 'market':
        portfoli_df['weight'] = portfoli_df.groupby(['t','portfolios'])['me'].transform(lambda x: x/sum(x))
        portfoli_df['ret'] = portfoli_df['ret'] * portfoli_df['weight']
    elif weighting == 'Equal':
        portfoli_df['weight'] = portfoli_df.groupby(['t','portfolios'])['me'].transform(lambda x: 1/len(x))
        portfoli_df['ret'] = portfoli_df['ret'] * portfoli_df['weight']
    return portfoli_df.groupby(['t','portfolios']).ret.sum().unstack().reset_index().rename(columns = {"t":"month"})
sorting_char = 'VolMkt'
#%% Changes from the original code
number_of_portfolios = 10
df = df.loc[df.prc >= 5].copy()
