#%%
import pandas as pd
# import statsmodels.formula.api as sm

import matplotlib.pyplot as plt
from arch import arch_model
gamma = 2
#%%
data = pd.read_excel("Assignment1Data_G1.xlsx", sheet_name="Predictability")
prediction_data = pd.read_excel("Out/prediction_data.xlsx")
prediction_data.drop(columns=[
    'DP',
    'OLS',
    'CM',
    'epsilon_tilde',
    'epsilon_hat_DP',
    'epsilon_hat_OLS',
    'epsilon_hat_CM',
    'omega_hat',
    'omega_hat_DP',
    'r_p_DP',
    'omega_hat_OLS',
    'r_p_OLS',
    'omega_hat_CM',
    'r_p_CM'], inplace=True)
prediction_data
#%% (ii)
def get_GARCH_var(data, model = 'GARCH'):
    garch = arch_model(100*data, mean='Constant', vol=model, p=1, q=1)
    res = garch.fit(show_warning = False, disp='off')
    forecast = res.forecast(horizon=1)
    return forecast.variance[-1:].values[0, 0]/(100**2)

prediction_data['GARCH_var'] = prediction_data.ExcessRet.expanding(min_periods=60).apply(get_GARCH_var, raw=True)

#%% 
prediction_data.rename(columns={
    "omega_hat_BM" : "omega_RW",
    "r_p_BM" : "r_p_RW",
}, inplace=True)

prediction_data['omega_hat_GARCH'] = prediction_data["BM"]/prediction_data['GARCH_var']/gamma
prediction_data.loc[prediction_data['omega_hat_GARCH'] >=2, 'omega_hat_GARCH'] = 2
prediction_data.loc[prediction_data['omega_hat_GARCH'] <=-1, 'omega_hat_GARCH'] = -1
prediction_data['r_p_'+'GARCH'] = prediction_data['ExcessRet'] + prediction_data['omega_hat_'+'GARCH']*prediction_data['ExcessRet']

portfolio = pd.DataFrame()
portfolio = prediction_data[['Month','r_p_RW','r_p_GARCH']].copy()
portfolio = portfolio.dropna()
portfolio += 1
portfolio['Month'] = portfolio['Month'].astype(str)
portfolio = portfolio.set_index('Month')
portfolio = portfolio.cumprod()
portfolio.plot()
plt.figure(figsize=(10,5))
plt.plot(portfolio,linewidth=1.5)
plt.legend(['RW','GARCH'])
ticket_values = list(range(0,len(portfolio),12))
ticket_values.append(len(portfolio))
plt.xticks(ticket_values,rotation=45)
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return of the Portfolios based on Different Variance Models")
plt.xlabel("Year-Month")
plt.savefig("Out/Ex6_C.png",dpi=300,bbox_inches='tight')
plt.savefig("Out/Ex6_C.pdf",dpi=300,bbox_inches='tight')
plt.show()

#%%
weights = pd.DataFrame()
weights = prediction_data[['Month','omega_RW','omega_hat_GARCH']].copy()
weights = weights.dropna()
weights['Month'] = weights['Month'].astype(str)
weights = weights.set_index('Month')

plt.plot(weights,linewidth=1.5)
plt.legend(['RW','GARCH'])
ticket_values = list(range(0,len(portfolio),24))
ticket_values.append(len(portfolio))
plt.xticks(ticket_values,rotation=45)
plt.title("Weights of OLS and CM")
plt.xlabel("Year-Month")
plt.savefig("Out/Ex6_D.png",dpi=300,bbox_inches='tight')
plt.savefig("Out/Ex6_D.pdf",dpi=300,bbox_inches='tight')
#%%
def utility(r,sigma, gamma):
    return r - 0.5*gamma*sigma**2
utility_list = {}
for i in ["RW", "GARCH"]:
    r = prediction_data['r_p_'+i].dropna().mean()
    sigma = prediction_data['r_p_'+i].dropna().std()
    utility_list[i] = utility(r,sigma, gamma)
    if i != "RW":
        utility_list[i] = utility_list[i] - utility_list["RW"]
    print("Utility of " + i + " = " + str(utility_list[i].round(4)))
    print("-"*15)