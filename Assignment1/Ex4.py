#%%
import pandas as pd

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