#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import statsmodels.api as sm
from functions import *
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
sorting_char = 'VolMkt'
#% Changes from the original code
number_of_portfolios = 10
df = df.loc[df.prc >= 5].copy()
#%%
vol_portfolios = get_portfolios(df, sorting_char, number_of_portfolios,size_exclusion=0.2)
vol_mkt_portfolios = get_portfolios(df, sorting_char, number_of_portfolios,weighting='market',size_exclusion=0.2)

# long-short equal weighted portfolio
vol_portfolios['long_short'] = sign_dict[sorting_char]*(vol_portfolios.iloc[:,-1] - vol_portfolios.iloc[:,1])

# long-short market weighted portfolio
vol_mkt_portfolios['long_short'] = sign_dict[sorting_char]*(vol_mkt_portfolios.iloc[:,-1] - vol_mkt_portfolios.iloc[:,1])

plot_df = vol_mkt_portfolios[['month','long_short']].copy()
plot_df['portfolio'] = 'Market Weighted'
tempt_df = vol_portfolios[['month','long_short']].copy()
tempt_df['portfolio'] = 'Equal Weighted'
plot_df = pd.concat([plot_df,tempt_df]).reset_index(drop=True)
del tempt_df
plot_df['month'] = pd.to_datetime(plot_df['month'], format='%Y%m')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("whitegrid")
plot_df['long_short'] = plot_df.groupby('portfolio')['long_short'].transform(lambda x: x.cumsum())
plot_df
plot_df = plot_df.melt(id_vars=['month','portfolio']).rename(columns = {'value':'Cumulative Return'}).drop(columns= 'portfolios')
plot_df
sns.lineplot(data=plot_df, x='month', y='Cumulative Return', hue='portfolio', palette="tab10",style="portfolio", 
markers=False, legend="full")
plt.title(f"Cumulative Return of Long-Short Portfolios")
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.legend(title='Portfolio')
plt.savefig(data_out + "/4_3.png")
plt.savefig(data_out + "/4_3.pdf", format='pdf', bbox_inches='tight')
#%%
test_results = {}
test_mkt_results = {}
for i in models:
    test_results[i] = time_series_regression(vol_portfolios,factors,models[i])
    test_mkt_results[i] = time_series_regression(vol_mkt_portfolios,factors,models[i])

pd.DataFrame(test_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/4_c1.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_mkt_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/4_c2.tex", float_format="%1.3f",column_format="lcc")
# %% (d)
pubYear, sampleStart, sampleEnd = sample_time_dict[sorting_char]


test_results = {}
test_out_results = {}
test_mkt_results = {}
test_mkt_out_results = {}
for i in models:
    test_results[i] = run_time_series_test(vol_portfolios, factors, models[i], sampleStart, sampleEnd)
    test_out_results[i] = run_time_series_test(vol_portfolios, factors, models[i], pubYear)
    test_mkt_results[i] = run_time_series_test(vol_mkt_portfolios, factors, models[i], sampleStart, sampleEnd)
    test_mkt_out_results[i] = run_time_series_test(vol_mkt_portfolios, factors, models[i], pubYear)

pd.DataFrame(test_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/4_d1_in.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_out_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/4_d1_out.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_mkt_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/4_d2_in.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_mkt_out_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/4_d2_out.tex", float_format="%1.3f",column_format="lcc")

# %%
