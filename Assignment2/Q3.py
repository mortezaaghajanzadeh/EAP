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
number_of_portfolios = 5
# %% (a)
print("Equal Weighted" )
vol_portfolios = get_portfolios(df, sorting_char, number_of_portfolios)
print(vol_portfolios.loc[:,1:].mean())
# %% (b)
vol_mkt_portfolios = get_portfolios(df, sorting_char, number_of_portfolios, 'market')
print("Market Weighted" )
print(vol_mkt_portfolios.loc[:,1:].mean())
pd.concat([vol_portfolios.loc[:,1:].mean().to_frame().T.rename(columns = {1:'Lowest', 2:'2', 3:'3', 4:'4', 5:'Highest'},index={0:'Equal Weighted'}), vol_mkt_portfolios.loc[:,1:].mean().to_frame().T.rename(columns = {1:'Lowest', 2:'2', 3:'3', 4:'4', 5:'Highest'},index={0:'Market Weighted'})]).to_latex(data_out + "/3_1.tex", float_format="%.4f")
#%%
plot_df = vol_portfolios.copy()
plot_df = plot_df.set_index('month')
plot_df = plot_df.cumsum()
plot_df.reset_index(inplace=True)
plot_df['month'] = pd.to_datetime(plot_df['month'], format='%Y%m')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("whitegrid")
plot_df = plot_df.melt(id_vars='month').rename(columns = {'portfolios':'Portfolio', 'value':'Cumulative Return'})
sns.lineplot(data=plot_df, x='month', y='Cumulative Return', hue='Portfolio', palette="tab10",style="Portfolio", markers=False, legend="full")
plt.title(f"Cumulative Return of {sorting_char} Portfolios")
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.legend(title='Portfolio')
plt.savefig(data_out + "/3_1.png")
plt.savefig(data_out + "/3_1.pdf", format='pdf', bbox_inches='tight')
plt.show()

plot_df = vol_mkt_portfolios.copy()
plot_df = plot_df.set_index('month')
plot_df = plot_df.cumsum()
plot_df.reset_index(inplace=True)
plot_df['month'] = pd.to_datetime(plot_df['month'], format='%Y%m')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("whitegrid")
plot_df = plot_df.melt(id_vars='month').rename(columns = {'portfolios':'Portfolio', 'value':'Cumulative Return'})
sns.lineplot(data=plot_df, x='month', y='Cumulative Return', hue='Portfolio', palette="tab10",style="Portfolio", markers=False, legend="full")
plt.title(f"Cumulative Return of {sorting_char} Portfolios")
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.legend(title='Portfolio')
plt.savefig(data_out + "/3_2.png")
plt.savefig(data_out + "/3_2.pdf", format='pdf', bbox_inches='tight')
plt.show()


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
plt.savefig(data_out + "/3_3.png")
plt.savefig(data_out + "/3_3.pdf", format='pdf', bbox_inches='tight')


#%%

# load factors
FF5_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="FF5")
CAR_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="Carhart")
FF5Car_factors = FF5_factors.merge(
    CAR_factors[['month','WML']], on='month', how='left'
)
HXZ_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="HXZ").drop(columns=['Mkt-Rf'])
factors = FF5Car_factors.merge(
    HXZ_factors,on='month'
)


# Time series regression
def time_series_regression(portfolios, factors, FactorModel):
    portfolios = portfolios.merge(factors, on='month', how='left')
    portfolios = portfolios.dropna()
    X = portfolios[FactorModel]
    X = sm.add_constant(X)
    Y = portfolios['long_short']
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    pvalues = model.pvalues
    betas = model.params
    return [betas.iloc[0],pvalues.iloc[0]]

CAPM_model = ['Mkt-Rf']
FF3_model = ['Mkt-Rf', 'SMB', 'HML']
CAR_model = ['Mkt-Rf', 'SMB', 'HML', 'WML']
FF5_model = ['Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA']
HXZ_model = ['Mkt-Rf', 'R_ME', 'R_IA', 'R_ROE']

models = {
    'CAPM' : ['Mkt-Rf'],
    'FF3' : ['Mkt-Rf', 'SMB', 'HML'],
    'CAR' : ['Mkt-Rf', 'SMB', 'HML', 'WML'],
    'FF5' : ['Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA'],
    'HXZ' : ['Mkt-Rf', 'R_ME', 'R_IA', 'R_ROE']
}

time_series_regression(vol_portfolios, factors, FF5_model)

test_results = {}
test_mkt_results = {}
for i in models:
    test_results[i] = time_series_regression(vol_portfolios,factors,models[i])
    test_mkt_results[i] = time_series_regression(vol_mkt_portfolios,factors,models[i])

pd.DataFrame(test_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/3_c1.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_mkt_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/3_c2.tex", float_format="%1.3f",column_format="lcc")



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

test_results = {}
test_out_results = {}
test_mkt_results = {}
test_mkt_out_results = {}
for i in models:
    test_results[i] = run_time_series_test(vol_portfolios, factors, models[i], sampleStart, sampleEnd)
    test_out_results[i] = run_time_series_test(vol_portfolios, factors, models[i], pubYear)
    test_mkt_results[i] = run_time_series_test(vol_mkt_portfolios, factors, models[i], sampleStart, sampleEnd)
    test_mkt_out_results[i] = run_time_series_test(vol_mkt_portfolios, factors, models[i], pubYear)

pd.DataFrame(test_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/3_d1_in.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_out_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/3_d1_out.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_mkt_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/3_d2_in.tex", float_format="%1.3f",column_format="lcc")
pd.DataFrame(test_mkt_out_results,index = [r'$\alpha$','$Pvalue$']).T.to_latex(data_out + "/3_d2_out.tex", float_format="%1.3f",column_format="lcc")






# %%
