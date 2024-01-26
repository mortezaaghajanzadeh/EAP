#%%
import pandas as pd
# import statsmodels.formula.api as sm
import statsmodels.api as sm
#%%
data = pd.read_excel("Assignment1Data_G1.xlsx", sheet_name="Predictability")
data = data.dropna()

# %%
print(data.shape)
data.head()
# %%
years = range(1970,2018)
periods = [int(str(i) + "0" + str(j)) for i in years for j in range(1,10) if len(str(j)) == 1]
periods.extend([int(str(i) + str(j)) for i in years for j in range(1,13) if len(str(j)) == 2 ])
periods.sort()
periods
# %% DP estimation
predection_period = 200105
in_sample_period = [i for i in periods if i < predection_period]
in_sample_data = data[data["Month"].isin(in_sample_period)]
# sm.ols(formula="ExcessRet ~ dp", data=in_sample_data).fit().params
X = sm.add_constant(in_sample_data["dp"])
y = in_sample_data["ExcessRet"]
beta = sm.OLS(y,X).fit().params.to_numpy()
X.iloc[-1].to_numpy() @ beta