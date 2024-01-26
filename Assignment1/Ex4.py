#%%
import pandas as pd
# import statsmodels.formula.api as sm
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
#%%
data = pd.read_excel("Assignment1Data_G1.xlsx", sheet_name="Predictability")
data = data.dropna()

# %%
print(data.shape)
data.head()
# %%
years = range(1970,2018)
periods = [int(str(i) + "0" + str(j)) for i in years for j in range(1,10) if len(str(j)) == 1]
periods.extend([int(str(i) + str(j)) for i in years for j in range(10,13) if len(str(j)) == 2 ])
periods.sort()
periods
# %% DP estimation

def prediction(X,y):
    beta = sm.OLS(y,X).fit().params.to_numpy()
    return X.iloc[-1].to_numpy() @ beta

DP_results = {}
OLS_results = {}
for prediction_period in tqdm([j for j in periods if j >= 198501]):
    in_sample_period = [i for i in periods if i < prediction_period]
    in_sample_data = data[data["Month"].isin(in_sample_period)]
    X = sm.add_constant(in_sample_data["dp"])
    y = in_sample_data["ExcessRet"]
    DP_results[prediction_period] = prediction(X,y)
    columns = list(data)
    columns.remove('Month')
    columns.remove('ExcessRet')
    columns.remove('Rfree')
    columns.remove('dp')
    X = sm.add_constant(in_sample_data[columns])
    OLS_results[prediction_period] = prediction(X,y)

# %%
