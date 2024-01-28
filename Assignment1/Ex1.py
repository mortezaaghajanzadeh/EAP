#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import t
import statsmodels.api as sm
#%%
#set seed 
np.random.seed(123)
# %% Reading the Data
df = pd.read_excel("Assignment1Data_G1.xlsx" , sheet_name="TwoStocks")
df.head()
#%% (a) compute the point estimates
mu_1 = sum(df['Stock1'])/len(df['Stock1'])
mu_2 = sum(df['Stock2'])/len(df['Stock2'])
print("The point estimates for the mean of stock 1 and stock 2 are {} and {} respectively".format(round(mu_1,4),round(mu_2,4)))
sigma_1 = sum((df.Stock1 - mu_1)**2)/(len(df.Stock1))
sigma_2 = sum((df.Stock2 - mu_2)**2)/(len(df.Stock2))
print("The point estimates for the variance of stock 1 and stock 2 are {} and {} respectively".format(round(sigma_1,4),round(sigma_2,4)))
# %% (b) compute the estimation covariance matrix
theta = np.array([mu_1,mu_2, sigma_1, sigma_2])

def f_v(theta,x):
    mu_1 = theta[0]
    mu_2 = theta[1]
    sigma_1 = theta[2]
    sigma_2 = theta[3]
    x_1 = x[0]
    x_2 = x[1]
    f = np.array([x_1-mu_1,x_2-mu_2,(x_1-mu_1)**2-sigma_1,(x_2-mu_2)**2-sigma_2]).reshape(len(theta),1)
    return f
def s(theta,x):
    f = f_v(theta,x)
    return f @ f.T
x = np.array([df[['Stock1','Stock2']]])[0]
s_hat = sum([s(theta,i) for i in x])/len(x)
# set non-diagonal elements to zero
s_hat = s_hat * np.eye(4)
print("The estimated standard error of estimation:\n {}".format(s_hat.round(4)))
#%% (c) compute the Newey-West standard errors
theta = np.array([mu_1,mu_2, sigma_1, sigma_2])
lag = 1
def gamma(theta,x,lag):
    gamma = {}
    for i in range(1,lag +1):
        lag = np.array(df[['Stock1','Stock2']].shift(i).dropna())
        tempt = []
        for num,j in enumerate(x[i:]):
            tempt.append(f_v(theta,j) @ f_v(theta,lag[num]).T)
        gamma[i] = sum(tempt)/len(tempt)
    gamma = [gamma[i] for i in gamma]
    return sum(gamma)

def s_newywest(theta,x):
    gamma_hat = gamma(theta,x,lag)
    return sum([s(theta,i) for i in x])/len(x) + 0.5 * (gamma_hat + gamma_hat.T)
s_hat_newywest = s_newywest(theta,x)
s_hat_newywest * np.eye(4)
#%% (d) compare the Sharpe ratio
R1 = df['Stock1'].values
R2 = df['Stock2'].values
sharpe_ratio_stock1 = np.mean(R1) / np.std(R1)
sharpe_ratio_stock2 = np.mean(R2) / np.std(R2)
print(f'Sharpe Ratio Stock 1: {sharpe_ratio_stock1:.4f}')
print(f'Sharpe Ratio Stock 2: {sharpe_ratio_stock2:.4f}')

R_theta = mu_1*sigma_2 - mu_2*sigma_1
R_prime = np.array([sigma_2, -sigma_1, -mu_2, mu_1]).reshape(4,1)
V_T = R_prime @ s_hat @ R_prime.T








#%%



# %%
# Perform t-test
t_stat, p_value = ttest_ind(R1, R2, equal_var=False)

print(f'T-test Statistic: {t_stat:.4f}')
print(f'p-value: {p_value:.4f}')

# Check for significance at a 5% level
if p_value < 0.05:
    print('Reject the null hypothesis: Stocks have different Sharpe ratios.')
else:
    print('Fail to reject the null hypothesis: No significant difference in Sharpe ratios.')
    
#%%
# adding covariance matrix assuming the normality and serial dependence   
correlation = np.corrcoef(R1, R2)[0, 1]
n1 = len(R1)
n2 = len(R2)

# Perform t-test
numerator = (sharpe_ratio_stock1 - sharpe_ratio_stock2) - 0  # Assuming no difference in Sharpe ratios under the null hypothesis
denominator = np.sqrt((np.std(R1)**2 / n1) + (np.std(R2)**2 / n2) - 2 * correlation * np.std(R1) * np.std(R2) / np.sqrt(n1 * n2))
t_stat = numerator / denominator
# Degrees of freedom
df = n1 + n2 - 2

# Calculate p-value
p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=df))

print(f'Sharpe Ratio Stock 1: {sharpe_ratio_stock1:.4f}')
print(f'Sharpe Ratio Stock 2: {sharpe_ratio_stock2:.4f}')
print(f'Sample Correlation: {correlation:.4f}')
print(f'T-test Statistic: {t_stat:.4f}')
print(f'Degrees of Freedom: {df}')
print(f'p-value: {p_value:.4f}')

# Check for significance at a 5% level
if p_value < 0.05:
    print('Reject the null hypothesis: Stocks have different Sharpe ratios.')
else:
    print('Fail to reject the null hypothesis: No significant difference in Sharpe ratios.')
    
# %%
#bootstrap method
# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns)

# Parameters
n_iterations = 10000
confidence_level = 0.95

# Calculate observed Sharpe ratio difference
sharpe_ratio_diff_observed = calculate_sharpe_ratio(R1) - calculate_sharpe_ratio(R2)

# Bootstrap procedure

sharpe_ratio_diff_bootstrap = []
np.random.seed(123)
for _ in range(n_iterations):
    # Generate bootstrapped samples
    bootstrap_sample_stock1 = np.random.choice(R1, size=len(R1), replace=True)
    bootstrap_sample_stock2 = np.random.choice(R2, size=len(R2), replace=True)

    # Calculate Sharpe ratios for bootstrapped samples
    sharpe_ratio_stock1 = calculate_sharpe_ratio(bootstrap_sample_stock1)
    sharpe_ratio_stock2 = calculate_sharpe_ratio(bootstrap_sample_stock2)

    # Store the Sharpe ratio difference
    sharpe_ratio_diff_bootstrap.append(sharpe_ratio_stock1 - sharpe_ratio_stock2)

# Calculate 95% confidence interval
lower_bound, upper_bound = np.percentile(sharpe_ratio_diff_bootstrap, [(1 - confidence_level) * 100 / 2, confidence_level * 100 - (1 - confidence_level) * 100 / 2])

print(f'Observed Sharpe Ratio Difference: {sharpe_ratio_diff_observed:.4f}')
print(f'95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]')
# The sharp ratio difference is not significant at 5% level.


# %%
#question 3 with 840 observations
# Set random seed for reproducibility
np.random.seed(123)

# Parameters
alpha = 0
beta = 0.2
theta = 0
phi = 0.9 
rho = 0.98
sigma_u = 0.05
sigma_v = 0.003
corr_uv = -0.98
n_obs = 840
n_replications = 10000

# Function to simulate data and estimate parameters
def simulate_and_estimate():
    # Initialize arrays to store parameter estimates
    beta_hat_array = np.zeros(n_replications)
    phi_hat_array = np.zeros(n_replications)

    for rep in range(n_replications):
        # Simulate data
        u = np.random.normal(0, sigma_u, n_obs)
        v = np.random.normal(0, sigma_v, n_obs)
        x = np.zeros(n_obs)
        r = np.zeros(n_obs)

        for t in range(1, n_obs):
            x[t] = theta + phi * x[t-1] + v[t]
            r[t] = alpha + beta * x[t] + u[t]

        # Estimate parameters with OLS
        model1 = sm.OLS(r, x)
        model2 = sm.OLS(x[1:], x[:-1])
        results1 = model1.fit()
        results2 = model2.fit()
        # Store estimates
        beta_hat_array[rep] = results1.params[0]
        phi_hat_array[rep] = results2.params[0]

    return beta_hat_array, phi_hat_array

# Perform Monte Carlo simulation
beta_hat_sim, phi_hat_sim = simulate_and_estimate()

# Plot histograms of parameter estimates
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(beta_hat_sim, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Beta Estimates')
plt.xlabel('Beta')

plt.subplot(1, 2, 2)
plt.hist(phi_hat_sim, bins=50, color='green', alpha=0.7)
plt.title('Histogram of Phi Estimates')
plt.xlabel('Phi')

plt.tight_layout()
plt.show()