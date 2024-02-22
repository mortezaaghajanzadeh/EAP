#%%
import pandas as pd
import numpy as np
from tabulate import tabulate
import statsmodels.api as sm
#%%
data_path = ''
#%%
df = pd.read_excel(data_path + 'Assignment3Data.xlsx')
df.describe()
df.set_index('month', inplace=True)
df = np.log(1 + df/100)
df.reset_index(inplace=True)
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
df.iloc[:,1:].describe()
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
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds= {'maxlags': int(len(Y)**0.25)}) 
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
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds= {'maxlags':int(len(Y)**0.25)}) 
    coef = model.params
    sd = model.bse
    return (coef, sd)

fama_bliss_regression(df, "24")

#%%
results = {}  # Create an empty dictionary to store the results
table_data_beta = []
table_data_alpha = []
for i in n:
    num_i = int(i)
    if num_i != 12 and num_i != 24:
        results[i] = {}  # Create a nested dictionary for each regression type
        results[i]['fama_bliss_regression'] = fama_bliss_regression(df, i)
        print(results[i]['fama_bliss_regression'][0])
        results[i]['campbell_shiller_regression'] = campbell_shiller_regression(df, i)
        results[i]['backus_regression'] = backus_regression(df, i)

        # Append the results to the table data
        
        table_data_alpha.append([
            i,
            results[i]['fama_bliss_regression'][0][0],
            results[i]['fama_bliss_regression'][1][0],
            results[i]['campbell_shiller_regression'][0][0],
            results[i]['campbell_shiller_regression'][1][0],
            results[i]['backus_regression'][0][0],
            results[i]['backus_regression'][1][0]
        ])
        table_data_beta.append([
            i,
            results[i]['fama_bliss_regression'][0][1],
            results[i]['fama_bliss_regression'][1][1],
            results[i]['campbell_shiller_regression'][0][1],
            results[i]['campbell_shiller_regression'][1][1],
            results[i]['backus_regression'][0][1],
            results[i]['backus_regression'][1][1]
        ])


# Define the table headers
headers = ['i', 'Fama-Bliss Coef', 'Fama-Bliss Std', 'Campbell-Shiller Coef', 'Campbell-Shiller Std', 'Backus Coef', 'Backus Std']

# Print the table
print(tabulate(table_data_beta, headers=headers))
# %%
