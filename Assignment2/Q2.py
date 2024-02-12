#%%
import pandas as pd
import numpy as np
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
    portfoli_df['portfolios'] = portfoli_df['portfolios'] + 1   # Highest is 5
    if weighting == 'market':
        portfoli_df['weight'] = portfoli_df.groupby(['t','portfolios'])['me'].transform(lambda x: x/sum(x))
        portfoli_df['ret'] = portfoli_df['ret'] * portfoli_df['weight']

    return portfoli_df.groupby(['t','portfolios']).ret.mean().unstack().reset_index().rename(columns = {"t":"month"})
sorting_char = 'VolMkt'
number_of_portfolios = 5
# %% (a)
print("Equal Weighted" )
vol_portfolios = get_portfolios(df, sorting_char, number_of_portfolios)
print(vol_portfolios.loc[:,1:].mean())
# %% (b)
vol_mkt_portfolios = get_portfolios(df, sorting_char, number_of_portfolios, 'market')
print("Market Weighted" )
print(vol_mkt_portfolios.loc[:,1:].mean())
# %% (c)
sign_dict = {
    'PayoutYield':1,
    'IdioVolAHT':-1,
    'RD':1,
    'AnnouncementReturn':1,
    'VolMkt':-1
    }
# long-short equal weighted portfolio
vol_portfolios['long_short'] = sign_dict[sorting_char]*(vol_portfolios[5] - vol_portfolios[1])

# long-short market weighted portfolio
vol_mkt_portfolios['long_short'] = sign_dict[sorting_char]*(vol_mkt_portfolios[5] - vol_mkt_portfolios[1])

# load factors
FF5_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="FF5")
HXZ_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="HXZ")

# Time series regression
def time_series_regression(portfolios, factors, FactorModel):
    portfolios = portfolios.merge(factors, on='month', how='left')
    portfolios = portfolios.dropna()
    X = portfolios[FactorModel]
    X = sm.add_constant(X)
    Y = portfolios['long_short']
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) # HAC standard errors
    # get pvalues
    pvalues = model.pvalues
    betas = model.params
    return pvalues.iloc[0]

CAPM_model = ['Mkt-RF']
FF3_model = ['Mkt-RF', 'SMB', 'HML']
FF5_model = ['Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA']
HXZ_model = ['Mkt-Rf', 'R_ME', 'R_IA', 'R_ROE']

time_series_regression(vol_portfolios, FF5_factors, FF5_model)

# %% (d)

sample_time_dict = {
    'PayoutYield': (2007,1984,2003),
    'IdioVolAHT': (2003,1976,1997),
    'RD': (2001,1975,1995),
    'AnnouncementReturn': (1996,1977,1992),
    'VolMkt': (1996,1979,1993)
}
# Perform the same time-series tests as in problem c above, using two sub-sample periods: 1) Sample period of the original paper and 2) Post-publication period

time_line = sample_time_dict[sorting_char]
pubYear = int(str(time_line[0]) + '01')
sampleStart = int(str(time_line[1]) + '01')
sampleEnd = int(str(time_line[2]) + '12')

def run_time_series_test(df, factors, FactorModel, sampleStart, sampleEnd=False):
    if sampleEnd != False:
        sample_df = df[(df['month'] >= sampleStart) & (df['month'] <= sampleEnd)]
    else:
        sample_df = df[(df['month'] >= sampleStart)]

    return time_series_regression(sample_df, factors, FactorModel)

print("Original Sample")
print(run_time_series_test(vol_mkt_portfolios, FF5_factors, FF5_model, sampleStart, sampleEnd))
print("Post Publication")
print(run_time_series_test(vol_mkt_portfolios, FF5_factors, FF5_model, pubYear))



