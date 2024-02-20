#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
#%%
data_path = ''
#%%
df = pd.read_excel(data_path + 'Assignment3Data.xlsx')
# %%
n = [i[1:] for i in list(df.columns[1:])]
for i in n:
    num_i = int(i)
    if num_i != 12:
        df['f'+i] = num_i /12 * df['y'+i] - (num_i/12 -1) * df['y'+str(num_i-12)]
        df['r'+i] = num_i /12 * df['y'+i].shift(12) - (num_i/12 -1) * df['y'+str(num_i-12)]
        df['rx'+i] = df['r'+i] - df['y12'].shift(12)     
df.head(13)
# %%
df.describe()
#%%
# FamaBliss 
def fama_bliss_regression(df, i):
    columns = ['rx'+i,'f'+i,'y12']
    columns = list(dict.fromkeys(columns))
    tempt_df = df[columns].dropna().copy()
    tempt_df['control'] = tempt_df['f'+i] - tempt_df['y12']
    X = tempt_df['control']
    X = sm.add_constant(X)
    Y = tempt_df['rx'+i]
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    coef = model.params
    sd = model.bse
    return (coef, sd)
# CampbellShiller
def campbell_shiller_regression(df, i):
    columns = ['y'+str(num_i-12),'y'+i,'y12']
    columns = list(dict.fromkeys(columns))
    tempt_df = df[columns].dropna().copy()
    tempt_df['control'] = (tempt_df['y'+i] - tempt_df['y12']) / (int(i)/12-1)
    tempt_df['Y'] = tempt_df['y'+str(num_i-12)].shift(-12) - tempt_df['y'+i]
    tempt_df = tempt_df.dropna()
    X = tempt_df['control']
    X = sm.add_constant(X)
    Y = tempt_df['Y']
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    coef = model.params
    sd = model.bse
    return (coef, sd)
# Backus
def backus_regression(df, i):
    columns = ['f'+str(num_i-12),'f'+i,'y12']
    columns = list(dict.fromkeys(columns))
    tempt_df = df[columns].dropna().copy()
    tempt_df['control'] = tempt_df['f'+i] - tempt_df['y12']
    tempt_df['Y'] = tempt_df['f'+str(num_i-12)].shift(-12) - tempt_df['y12']
    tempt_df = tempt_df.dropna()
    X = tempt_df['control']
    X = sm.add_constant(X)
    Y = tempt_df['Y']
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    coef = model.params
    sd = model.bse
    return (coef, sd)
#%%
results = {}  # Create an empty dictionary to store the results

for i in n:
    print(i)
    num_i = int(i)
    if num_i != 12 and num_i != 24:
        results[i] = {}  # Create a nested dictionary for each regression type
        results[i]['fama_bliss_regression'] = fama_bliss_regression(df, i)
        results[i]['campbell_shiller_regression'] = campbell_shiller_regression(df, i)
        results[i]['backus_regression'] = backus_regression(df, i)

# Print the results
for i, reg_results in results.items():
    print(f'Results for {i}:')
    print('fama_bliss_regression', reg_results['fama_bliss_regression'])
    print('campbell_shiller_regression', reg_results['campbell_shiller_regression'])
    print('backus_regression', reg_results['backus_regression'])
# %%
