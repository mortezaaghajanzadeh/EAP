#%%
import pandas as pd
import numpy as np
data_path = r"Data"
data_out = r"Out"
#%%
consumption_df = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx",sheet_name="Consumption")
consumption_df

#%% (a)
# % The means, standard deviations, and first-order autocorrelation coefficients

# function to calculate the mean, standard deviation and first-order autocorrelation coefficient
def mean_std_autocorr(df,column_name):
    g = df[column_name].copy()
    mean = g.mean()
    std = g.std()
    autocorr_1 = g.autocorr(lag=1)
    return mean, std, autocorr_1

# create a excess return column
consumption_df['Re'] = consumption_df['Market'] - consumption_df['Rfree']

estimation_results = {}
# apply the function to the consumption data
for i in ['ConsGrowth', 'Market', 'Rfree', 'Re']:
    mean, std, autocorr_1 = mean_std_autocorr(consumption_df,i)
    print(f"{i}: mean = {mean: .3f}, std = {std: .3f}, autocorr_1 = {autocorr_1: .3f}")
    estimation_results[i] = [mean, std, autocorr_1]

pd.DataFrame(estimation_results, index=['$\mu$', '$\sigma$', r'$\rho_1$']).rename(columns={'ConsGrowth': '$\Delta c$', 'Market': '$r_{m,t}$', 'Rfree': '$r_{f,t}$', 'Re': '$r_{e,t}$'}).to_latex(data_out + "/1a.tex", float_format="%.3f",)
#% Sample correlation matrix

# calculate the correlation matrix
corr_df = consumption_df[['ConsGrowth', 'Market', 'Rfree', 'Re']].cov()

corr_df.rename(index={'ConsGrowth': '$\Delta c$', 'Market': '$r_{m,t}$', 'Rfree': '$r_{f,t}$', 'Re': '$r_{e,t}$'}, columns={'ConsGrowth': '$\Delta c$', 'Market': '$r_{m,t}$', 'Rfree': '$r_{f,t}$', 'Re': '$r_{e,t}$'}).to_latex(data_out + "/1a_corr.tex", float_format="%.3f",)
# %% (b)
# The first estimation:
gamma_1 = (
    estimation_results['Re'][0] + estimation_results['Market'][1] ** 2 /2
    ) / corr_df.loc['ConsGrowth', 'Market']
print(f"The first estimation: gamma = {gamma_1: .3f}")

# The second estimation:
gamma_2 = (
    estimation_results['Re'][0] + estimation_results['Market'][1] ** 2 /2
    ) / estimation_results['Market'][1]/estimation_results['ConsGrowth'][1]
print(f"The second estimation: gamma = {gamma_2: .3f}",)
# %% (c)
delta_1 = np.exp(-1 * (estimation_results['Rfree'][0] - gamma_1 * estimation_results['ConsGrowth'][0] + 0.5 * gamma_1**2 * estimation_results['ConsGrowth'][1]**2 ))

delta_2 = np.exp(-1 * (estimation_results['Rfree'][0] - gamma_2 * estimation_results['ConsGrowth'][0] + 0.5 * gamma_2**2 * estimation_results['ConsGrowth'][1]**2 ))

print(f"The estimation of the discount factor with gamma_1: delta = {delta_1: .3f} and with gamma_2: delta = {delta_2: .3f}")

print(f"The estimation of the time preference rate with gamma_1: rho = {-np.log(delta_1): .3f} and with gamma_2: rho = {-np.log(delta_2): .3f}")

# %% (d)
mu_c = estimation_results['ConsGrowth'][0]
mu_m = estimation_results['Market'][0]
gamma = gamma_1
delta = delta_1
theta = np.array([mu_c, mu_m, gamma, delta])

def f_v(theta,x):
    mu_c = theta[0]
    mu_m = theta[1]
    gamma = theta[2]
    delta = theta[3]
    x_c = x[0]
    x_m = x[1]
    x_e = x[2]
    r_f = x_m - x_e
    f = np.array([
        x_c - mu_c,
        x_m - mu_m,
        x_e + 0.5 * (x_m - mu_m)**2 - gamma * (x_m - mu_m) * (x_c - mu_c),
        r_f + np.log(delta) - gamma * x_c + 0.5 * gamma**2 * (x_c - mu_c)**2
        ]).reshape(len(theta),1)
    return f
def s(theta,x):
    f = f_v(theta,x)
    return f @ f.T

x = np.array([consumption_df[['ConsGrowth','Market','Re']]])[0]

x_df = consumption_df[['ConsGrowth','Market','Re']].copy()


lag = 1
def gamma(theta,x,lag,x_df):
    gamma = {}
    for i in range(1,lag +1):
        lag = np.array(x_df.shift(i).dropna())
        tempt = []
        for num,j in enumerate(x[i:]):
            tempt.append(f_v(theta,j) @ f_v(theta,lag[num]).T)
        gamma[i] = sum(tempt)/len(tempt)
    gamma = [gamma[i] for i in gamma]
    return sum(gamma)

def s_newywest(theta,x,lag,x_df):
    gamma_hat = gamma(theta,x,lag,x_df)
    return sum([s(theta,i) for i in x])/len(x) + 0.5 * (gamma_hat + gamma_hat.T)
s_hat_newywest = s_newywest(theta,x,lag,x_df)
pd.DataFrame(s_hat_newywest, columns =['$\mu_c$', '$\mu_m$', '$\gamma$', '$\delta$'], index =['$\mu_c$', '$\mu_m$', '$\gamma$', '$\delta$']).to_latex(data_out + "/1d.tex", float_format="%.4f")

# %% (e)

def f_v(theta,x):
    gamma = theta[0]
    delta = theta[1]
    x_c = x[0]
    x_m = x[1]
    x_e = x[2]
    r_f = x_m - x_e
    f = np.array([
        np.exp(np.log(delta) - gamma * x_c + x_m) - 1,
        np.exp(np.log(delta) - gamma * x_c + r_f) - 1,
        ]).reshape(len(theta),1)
    return f

def FOC(theta,x):
    return sum([f_v(theta,i) for i in x])/len(x)

def loss(theta,x):
    return (FOC(theta,x).T @ FOC(theta,x))[0][0]


import numpy as np
from scipy.optimize import minimize
first_guess = np.array([gamma_1, delta_1])
res = minimize(loss, first_guess, args=(x), method='BFGS', options={'disp': True})

pd.DataFrame(res.x,columns = [r'$\hat{\theta}$'],index = ['$\gamma$',"$\delta$"]).T.to_latex(data_out + "/1e.tex", float_format="%.2f")
print(f"The estimation of the gamma with the new method: gamma = {res.x[0]: .3f} and with the old method: gamma = {gamma_1: .3f}")
print(f"The estimation of the delta with the new method: delta = {res.x[1]: .3f} and with the old method: delta = {delta_1: .3f}")

sd_newywest = s_newywest(res.x,x,lag,x_df)
pd.DataFrame(sd_newywest,columns =  ['$\gamma$',"$\delta$"],index = ['$\gamma$',"$\delta$"]).to_latex(data_out + "/1e_sd.tex", float_format="%.4f")
pd.DataFrame(sd_newywest,columns =  ['gamma',"delta"],index = ['gamma',"delta"])
