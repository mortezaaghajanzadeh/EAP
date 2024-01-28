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
# %%
#question 1
df = pd.read_excel("Assignment1Data_G1.xlsx" , sheet_name="TwoStocks")
R1 = df['Stock1'].values
R2 = df['Stock2'].values
sharpe_ratio_stock1 = np.mean(R1) / np.std(R1)
sharpe_ratio_stock2 = np.mean(R2) / np.std(R2)

# %%
# Perform t-test
t_stat, p_value = ttest_ind(R1, R2, equal_var=False)

print(f'Sharpe Ratio Stock 1: {sharpe_ratio_stock1:.4f}')
print(f'Sharpe Ratio Stock 2: {sharpe_ratio_stock2:.4f}')
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

# %%
#question 3 with 240 observations
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
n_obs = 240
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
# %%
#(b)
np.random.seed(123)
# Parameters
alpha = 0
beta = 0
theta = 0
phi = 0
rho = 0.98
sigma_u = 0.05
sigma_v = 0.003
sigma_uv = -0.98
num_simulations = 10000  # Number of Monte Carlo simulations
sample_size = 840 # Sample size for each simulation
horizon = 12            # Horizon for long-horizon predictability regression
significance_level = 0.05
reject_count = 0
for _ in range(num_simulations):
    # Simulate the system
    u = np.random.normal(0, sigma_u, sample_size)
    v = np.random.normal(0, sigma_v, sample_size)
    x = np.zeros(sample_size)
    r = np.zeros(sample_size)

    for t in range(sample_size - 1):
        x[t + 1] = theta + phi * x[t] + v[t + 1]
        r[t + 1] = alpha + beta * x[t] + u[t + 1]
    
    rt_t_12 = np.zeros(sample_size - horizon + 1)  # Assuming the length should be sample_size - horizon + 1

    for t in range(sample_size - horizon + 1):
        rt_t_12[t] = np.sum(r[t:t + horizon])
    

    model = sm.OLS(rt_t_12, x[horizon - 1:])


    results = model.fit()

    # Test the null hypothesis H0: betaK = 0
    p_value = results.pvalues

    if p_value < significance_level:
        reject_count += 1

# Print the rejection rate
rejection_rate = reject_count / num_simulations
print(f'Rejection rate at {significance_level * 100}% significance level: {rejection_rate * 100}%')
#475 out of 10,000 are rejected, meaning that the null hypothesis is rejected at 4.75% significance level.
# %%
#(c)use the Newey and West (1987) standard errors of βˆK with a maximum lag at 11 when testing H0
np.random.seed(123)
# Parameters
alpha = 0
beta = 0
theta = 0
phi = 0
rho = 0.98
sigma_u = 0.05
sigma_v = 0.003
sigma_uv = -0.98
num_simulations = 10000  # Number of Monte Carlo simulations
sample_size = 840 # Sample size for each simulation
horizon = 12            # Horizon for long-horizon predictability regression
significance_level = 0.05
reject_count = 0
for _ in range(num_simulations):
    # Simulate the system
    u = np.random.normal(0, sigma_u, sample_size)
    v = np.random.normal(0, sigma_v, sample_size)
    x = np.zeros(sample_size)
    r = np.zeros(sample_size)

    for t in range(sample_size - 1):
        x[t + 1] = theta + phi * x[t] + v[t + 1]
        r[t + 1] = alpha + beta * x[t] + u[t + 1]
    
    rt_t_12 = np.zeros(sample_size - horizon + 1)  # Assuming the length should be sample_size - horizon + 1

    for t in range(sample_size - horizon + 1):
        rt_t_12[t] = np.sum(r[t:t + horizon])
    

    model = sm.OLS(rt_t_12, x[horizon - 1:])
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 11})


    # Test the null hypothesis H0: betaK = 0
    p_value = results.pvalues

    if p_value < significance_level:
        reject_count += 1

# Print the rejection rate
rejection_rate = reject_count / num_simulations
print(f'Rejection rate at {significance_level * 100}% significance level: {rejection_rate * 100}%')
#512 out of 10,000 are rejected, meaning that the null hypothesis is rejected at 5.12% significance level.

# %%
