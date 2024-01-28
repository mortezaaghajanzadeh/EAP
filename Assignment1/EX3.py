#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Parameters
alpha = 0
beta = 0.2
theta = 0
phi = 0.98
sigma_u = 0.05
sigma_v = 0.003
corr_uv = -0.98
n_obs = 840
n_replications = 10000

#%% Function to simulate data and estimate parameters
def simulate_and_estimate():
    # Initialize arrays to store parameter estimates
    beta_hat_array = np.zeros(n_replications)
    phi_hat_array = np.zeros(n_replications)

    for rep in range(n_replications):
        beta_hat_array[rep], phi_hat_array[rep] = simulate()

    return beta_hat_array, phi_hat_array

def simulate():
    r, x = generate_data()
        # Estimate parameters with OLS
    beta, phi = estimate_parameters(r, x)
    return beta, phi

def generate_data():
    # Simulate data
    # Correlation matrix
    corr_mat= np.array([[1.0, corr_uv],
                        [corr_uv, 1.0]])

    # Compute the (upper) Cholesky decomposition matrix
    upper_chol = cholesky(corr_mat)

    # Generate 3 series of normally distributed (Gaussian) numbers
    rnd = np.random.normal(0.0, 1.0, size=(n_obs, 2))

    # Finally, compute the inner product of upper_chol and rnd
    errors = rnd @ upper_chol
    u, v = errors[:, 0], errors[:, 1]
    # Scale errors
    u *= sigma_u
    v *= sigma_v 

    x = np.zeros(n_obs)
    r = np.zeros(n_obs)
    for t in range(1, n_obs):
        x[t] = theta + phi * x[t-1] + v[t]
        r[t] = alpha + beta * x[t-1] + u[t]
    return r, x

def estimate_parameters(r, x):
    # Estimate parameters with OLS
    model1 = sm.OLS(r, sm.add_constant(x))
    model2 = sm.OLS(x[1:], sm.add_constant(x[:-1]))
    results1 = model1.fit()
    results2 = model2.fit()
    # Store estimates
    beta_hat = results1.params[1]
    phi_hat = results2.params[1]
    return beta_hat, phi_hat


def plot_histograms(beta_hat_sim, phi_hat_sim,name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    # plt.hist(beta_hat_sim, bins=50, color='blue', alpha=0.7)
    ax = sns.histplot(beta_hat_sim, bins=100, color='blue', alpha=0.7, kde=True, stat='density', linewidth=0.1)
    ax.axvline(beta_hat_sim.mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title(r'Distribution of $\hat{\beta}$')
    plt.xlabel(r'$\beta$')

    plt.subplot(1, 2, 2)
    # plt.hist(phi_hat_sim, bins=50, color='green', alpha=0.7)
    ax = sns.histplot(phi_hat_sim, bins=50, color='green', alpha=0.7, kde=True, stat='density', linewidth=0.1)
    ax.axvline(phi_hat_sim.mean(), color='red', linestyle='dashed', linewidth=1)
    plt.title(r'Distribution of $\hat{\phi}$')
    plt.xlabel(r'$\phi$')
    plt.ylabel("")

    plt.tight_layout()
    #save pdf and png
    plt.savefig(f'Out/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'Out/{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Perform Monte Carlo simulation
np.random.seed(123)
beta_hat_sim, phi_hat_sim = simulate_and_estimate()

#Plot histograms of parameter estimates
plot_histograms(beta_hat_sim, phi_hat_sim,'EX3-1')

n_obs = 240
np.random.seed(123)
beta_hat_sim, phi_hat_sim = simulate_and_estimate()
plot_histograms(beta_hat_sim, phi_hat_sim,'EX3-2')
# %% (b)
beta = 0
n_obs = 840
def estimate_new_model(r,x):
    df = pd.DataFrame({'r':r,'x':x})
    df['12_month_return'] = df['r'].rolling(12).sum()
    # get the pvalue of the regression
    return sm.OLS(df['12_month_return'][11:], sm.add_constant(df['x'][11:])).fit().pvalues[1]
def new_simulation():
    pvalue = np.zeros(n_replications)
    for rep in range(n_replications):
        r, x = generate_data()
        pvalue[rep] = estimate_new_model(r,x)
    return pvalue
np.random.seed(123)
pvalue = new_simulation()
[np.mean(pvalue<0.05),np.mean(pvalue<0.01),np.mean(pvalue<0.001)]
#%% (c)
def estimate_new_model(r,x):
    df = pd.DataFrame({'r':r,'x':x})
    df['12_month_return'] = df['r'].rolling(12).sum()
    # get the pvalue of the regression
    return sm.OLS(df['12_month_return'][11:], sm.add_constant(df['x'][11:])).fit(cov_type='HAC', cov_kwds={'maxlags': 11}).pvalues[1]
def new_simulation():
    pvalue = np.zeros(n_replications)
    for rep in range(n_replications):
        r, x = generate_data()
        pvalue[rep] = estimate_new_model(r,x)
    return pvalue
np.random.seed(123)
pvalue = new_simulation()
[np.mean(pvalue<0.05),np.mean(pvalue<0.01),np.mean(pvalue<0.001)]
