#%%
from functions import *
import seaborn as sns

#%%
data_path = r"Data"
data_out = r"Out"
#%%
df = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="CrossSection")
df = df.merge(factors, on='month', how='left')
# %%
# run time series regression

results = {}
for i in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']:
    X = df['Mkt-Rf']
    X = sm.add_constant(X)
    Y = df[i]
    model = sm.OLS(Y, X)
    res_homo = model.fit()
    # get t-stats
    t_stats_normal = res_homo.tvalues
    betas = res_homo.params
    # Newey-West standard errors
    res_newey = model.fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    t_stats_newey = res_newey.tvalues
    results[i] = [betas.iloc[0], t_stats_normal.iloc[0], t_stats_newey.iloc[0]]
pd.DataFrame(results,index = [r'$\alpha$','$t_{homo}$','$t_{Newey}$']).T


