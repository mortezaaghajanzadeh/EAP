#%%
import pandas as pd
# import statsmodels.formula.api as sm
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
# %% Estimation
def prediction(X,y):
    beta = sm.OLS(y,X).fit().params.to_numpy()
    return X.iloc[-1].to_numpy() @ beta
BM_results = {}
DP_results = {}
OLS_results = {}
CM_results = {}
for prediction_period in tqdm([j for j in periods if j >= 198501]):
    in_sample_period = [i for i in periods if i < prediction_period]
    in_sample_data = data[data["Month"].isin(in_sample_period)]
    X = sm.add_constant(in_sample_data["dp"])
    y = in_sample_data["ExcessRet"]
    BM_results[prediction_period] = y.mean()
    DP_results[prediction_period] = prediction(X,y)
    columns = list(data)
    columns.remove('Month')
    columns.remove('ExcessRet')
    columns.remove('Rfree')
    columns.remove('dp')
    X = sm.add_constant(in_sample_data[columns])
    OLS_results[prediction_period] = prediction(X,y)
    CM_list= []
    for i in columns:
        X = sm.add_constant(in_sample_data[i])
        CM_list.append(prediction(X,y))
    CM_results[prediction_period] = np.mean(CM_list)

# %%
prediction_data = pd.DataFrame()
prediction_data = data[['Month', 'ExcessRet', 'Rfree']].copy()
labels = ["BM","DP", "OLS", "CM"]
for num,i in enumerate([BM_results,DP_results, OLS_results, CM_results]):
    prediction_data[labels[num]] = prediction_data["Month"].map(i)
prediction_data['epsilon_tilde'] = prediction_data['ExcessRet'] - prediction_data['BM']
for i in ["DP", "OLS", "CM"]:
    prediction_data['epsilon_hat_' + i] = prediction_data['ExcessRet'] - prediction_data[i]
prediction_data

R_squared_os = {}
for i in ["DP", "OLS", "CM"]:
    sum_epsilon_hat = np.sum(prediction_data['epsilon_hat_' + i]**2)
    sum_epsilon_tilde = np.sum(prediction_data['epsilon_tilde']**2)
    R_squared_os[i] = 1 - sum_epsilon_hat/sum_epsilon_tilde
    print("R_squared_os_" + i + " = " + str(R_squared_os[i].round(4)))

#%% (b) Diebold and Mariano test
def DM_test(y_tilde, y_hat):
    T = len(y_hat)
    d = y_tilde**2 - y_hat**2
    delta_hat = np.mean(d)
    # sigma_hat = np.sqrt(np.sum((d - delta_hat)**2)/(T-1))
    # Newey-West correction with on lag
    sigma_hat = np.sqrt(np.sum((d - delta_hat)**2)/(T-1) + 2*np.sum([d[i]*d[i-1] for i in range(1,T)])/(T-1))

    DM = delta_hat/sigma_hat * np.sqrt(T)
    return DM
def Clark_West_test(y_tilde, y_hat, R_tilde, R_hat):
    T = len(y_hat)
    d = y_tilde**2 - (y_hat**2 - (R_tilde - R_hat)**2)
    delta_hat = np.mean(d)
    # sigma_hat = np.sqrt(np.sum((d - delta_hat)**2)/(T-1))
    sigma_hat = np.sqrt(np.sum((d - delta_hat)**2)/(T-1) + 2*np.sum([d[i]*d[i-1] for i in range(1,T)])/(T-1))
    CW = delta_hat/sigma_hat * np.sqrt(T)
    return CW

y_tilde = prediction_data['epsilon_tilde'].dropna().to_numpy()
R_tilde = prediction_data['BM'].dropna().to_numpy()
DM_test_results = {}
CW_test_results = {}
for i in ["DP", "OLS", "CM"]:
    y_hat = prediction_data['epsilon_hat_' + i].dropna().to_numpy()
    R_hat = prediction_data[i].dropna().to_numpy()
    DM_test_results[i] = DM_test(y_tilde, y_hat)
    CW_test_results[i] = Clark_West_test(y_tilde, y_hat, R_tilde, R_hat)
    print("DM_" + i + " = " + str(DM_test_results[i].round(4)))
    print("CW_" + i + " = " + str(CW_test_results[i].round(4)))
    print("-"*15)

#%% (C)
prediction_data['rolling_var'] = prediction_data.ExcessRet.rolling(60).var()
gamma = 2
for prediction_model in ['BM',"DP", "OLS", "CM"]:
    prediction_data['omega_hat'] = prediction_data[prediction_model]/prediction_data['rolling_var']/gamma
    prediction_data['omega_hat_'+prediction_model] = prediction_data['omega_hat']
    prediction_data.loc[prediction_data.omega_hat >= 2,'omega_hat_'+prediction_model] = 2
    prediction_data.loc[prediction_data.omega_hat <= -1,'omega_hat_'+prediction_model] = -1
    prediction_data['r_p_'+prediction_model] = prediction_data['ExcessRet'] + prediction_data['omega_hat_'+prediction_model]*prediction_data['ExcessRet']
prediction_data.to_excel("Out/prediction_data.xlsx",index=False)
portfolio = pd.DataFrame()
portfolio = prediction_data[['Month','r_p_BM','r_p_DP','r_p_OLS','r_p_CM']].copy()
portfolio = portfolio.dropna()
portfolio['Month'] = portfolio['Month'].astype(str)
portfolio = portfolio.set_index('Month')
portfolio += 1
portfolio = portfolio.cumprod()
plt.figure(figsize=(10,5))
plt.plot(portfolio,linewidth=1.5)
plt.legend(['BM','DP','OLS','CM'])
ticket_values = list(range(0,len(portfolio),12))
ticket_values.append(len(portfolio))
plt.xticks(ticket_values,rotation=45)
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return of the Portfolios based on Different Prediction Models")
plt.xlabel("Year-Month")
plt.show()

# %% (d) plot the weights
weights = pd.DataFrame()
weights = prediction_data[['Month','omega_hat_BM','omega_hat_DP','omega_hat_OLS','omega_hat_CM']].copy()
weights = weights.dropna()
weights['Month'] = weights['Month'].astype(str)
weights = weights.set_index('Month')
weights.plot()
# %% (e) Investor's utility
def utility(r,sigma, gamma):
    return r - 0.5*gamma*sigma**2
utility_list = {}
for i in ["BM", "DP", "OLS", "CM"]:
    r = prediction_data['r_p_'+i].dropna().mean()
    sigma = prediction_data['r_p_'+i].dropna().std()
    utility_list[i] = utility(r,sigma, gamma)
    if i != "BM":
        print("The gain from the the " + i + " model = " + str((utility_list[i] - utility_list["BM"]).round(4)))
