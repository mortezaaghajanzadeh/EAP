#%%
import pandas as pd
import numpy as np
from tabulate import tabulate
import statsmodels.api as sm
#%%
data_path = 'data/'
out_path = 'out/'
#%%
df = pd.read_excel(data_path + 'Assignment3Data.xlsx')
df = df.set_index('month')
df = np.log(1 + df/100)
df.reset_index(inplace=True)
df.describe()
# %%
n = [i[1:] for i in list(df.columns[1:])]
yield_columns = ['y'+i for i in n]
forward_columns = []
log_return_columns = []
excess_return_columns = []
for i in n:
    num_i = int(i)
    if num_i != 12:
        df['f'+i] = num_i /12 * df['y'+i] - (num_i/12 -1) * df['y'+str(num_i-12)]
        df['r'+i] = num_i /12 * df['y'+i].shift(12) - (num_i/12 -1) * df['y'+str(num_i-12)]
        df['rx'+i] = df['r'+i] - df['y12'].shift(12)  
        forward_columns.append('f'+i)
        log_return_columns.append('r'+i)
        excess_return_columns.append('rx'+i)
df.head(13)
# %%
res_1 = df.iloc[:,1:].describe().drop(['count'])
res_1[yield_columns].to_latex(out_path + 'yield_describe.tex', float_format="%.2f")
res_1[forward_columns].to_latex(out_path + 'forward_describe.tex', float_format="%.2f")
res_1[log_return_columns].to_latex(out_path + 'log_return_describe.tex', float_format="%.2f")
res_1[excess_return_columns].to_latex(out_path + 'excess_return_describe.tex', float_format="%.2f")
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
    tstat = model.tvalues
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
    if num_i-12 == 12:
        df['f12'] = df['y12']

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
    sd = model.bse * (np.sqrt(len(Y)))
    return (coef, sd)

fama_bliss_regression(df, "36")

#%%
results = {}  # Create an empty dictionary to store the results
table_data_fama_bliss = []
table_data_campbell_shiller = []
table_data_backus = []
for i in n:
    num_i = int(i)
    if num_i != 12:
        results[i] = {}  # Create a nested dictionary for each regression type
        results[i]['fama_bliss_regression'] = fama_bliss_regression(df, i)
        results[i]['campbell_shiller_regression'] = campbell_shiller_regression(df, i)
        results[i]['backus_regression'] = backus_regression(df, i)

        # Create table data for fama_bliss_regression
        coef_fama_bliss, sd_fama_bliss = results[i]['fama_bliss_regression']
        table_data_fama_bliss.append([i, coef_fama_bliss[1], sd_fama_bliss[1], coef_fama_bliss[1]/sd_fama_bliss[1]])

        # Create table data for campbell_shiller_regression
        coef_campbell_shiller, sd_campbell_shiller = results[i]['campbell_shiller_regression']
        table_data_campbell_shiller.append([i, coef_campbell_shiller[1], sd_campbell_shiller[1], (coef_campbell_shiller[1] - 1)/sd_campbell_shiller[1]])

        # Create table data for backus_regression
        coef_backus, sd_backus = results[i]['backus_regression']
        table_data_backus.append([i, coef_backus[1], sd_backus[1], (coef_backus[1]-1)/sd_backus[1]])

# Create tables
table_fama_bliss = pd.DataFrame(table_data_fama_bliss, columns=['Maturity', 'Coefficient', 'Standard Deviation', 'Test stats']).set_index('Maturity')
table_campbell_shiller = pd.DataFrame(table_data_campbell_shiller, columns=['Maturity', 'Coefficient', 'Standard Deviation', 'Test stats']).set_index('Maturity')
table_backus = pd.DataFrame(table_data_backus, columns=['Maturity', 'Coefficient', 'Standard Deviation', 'Test stats']).set_index('Maturity')

# to latex
table_fama_bliss.to_latex(out_path + 'fama_bliss.tex', float_format="%.2f",column_format="lcccc",index=True)

table_campbell_shiller.to_latex(out_path + 'campbell_shiller.tex', float_format="%.2f",column_format="lcccc",index=True)

table_backus.to_latex(out_path + 'backus.tex', float_format="%.2f",column_format="lcccc",index=True)

# Print tables
print("Fama Bliss Regression Results (Control):")
print(table_fama_bliss)

print("Campbell Shiller Regression Results (Control):")
print(table_campbell_shiller)

print("Backus Regression Results (Control):")
print(table_backus)




# %%
