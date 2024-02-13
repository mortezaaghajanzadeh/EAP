import pandas as pd
import statsmodels.api as sm
data_path = r"Data"
data_out = r"Out"
def get_portfolios(df, sorting_car, number_of_portfolios,weighting = 'Equal',size_exclusion = False):
    portfoli_df = df.dropna(subset=[sorting_car])[
        ['t', 'permno', sorting_car, 'me', 'ret']
    ].copy()
    if size_exclusion != False:
        portfoli_df = portfoli_df.loc[portfoli_df.me >= portfoli_df.me.quantile(size_exclusion)].copy()

    portfoli_df['portfolios'] = portfoli_df.groupby('t')[sorting_car].transform(lambda x: pd.qcut(x, number_of_portfolios, labels=False)) 
    portfoli_df['portfolios'] = portfoli_df['portfolios'] + 1   # The highest value is the highest portfolio
    if weighting == 'market':
        portfoli_df['weight'] = portfoli_df.groupby(['t','portfolios'])['me'].transform(lambda x: x/sum(x))
        portfoli_df['ret'] = portfoli_df['ret'] * portfoli_df['weight']
    elif weighting == 'Equal':
        portfoli_df['weight'] = portfoli_df.groupby(['t','portfolios'])['me'].transform(lambda x: 1/len(x))
        portfoli_df['ret'] = portfoli_df['ret'] * portfoli_df['weight']
    return portfoli_df.groupby(['t','portfolios']).ret.sum().unstack().reset_index().rename(columns = {"t":"month"})




def time_series_regression(portfolios, factors, FactorModel):
    portfolios = portfolios.merge(factors, on='month', how='left')
    portfolios = portfolios.dropna()
    X = portfolios[FactorModel]
    X = sm.add_constant(X)
    Y = portfolios['long_short']
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':int(len(Y)**0.25)}) 
    pvalues = model.pvalues
    betas = model.params
    return [betas.iloc[0],pvalues.iloc[0]]



sample_time_dict = {
    'PayoutYield': (200701,198401,200312),
    'IdioVolAHT': (200301,197601,199712),
    'RD': (200101,197501,199512),
    'AnnouncementReturn': (199601,197701,199212),
    'VolMkt': (199601,197901,199312)
}



def run_time_series_test(df, factors, FactorModel, sampleStart, sampleEnd=False):
    if sampleEnd != False:
        sample_df = df[(df['month'] >= sampleStart) & (df['month'] <= sampleEnd)]
    else:
        sample_df = df[(df['month'] >= sampleStart)]

    return time_series_regression(sample_df, factors, FactorModel)

# Inputs
CAPM_model = ['Mkt-Rf']
FF3_model = ['Mkt-Rf', 'SMB', 'HML']
CAR_model = ['Mkt-Rf', 'SMB', 'HML', 'WML']
FF5_model = ['Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA']
HXZ_model = ['Mkt-Rf', 'R_ME', 'R_IA', 'R_ROE']

models = {
    'CAPM' : ['Mkt-Rf'],
    'FF3' : ['Mkt-Rf', 'SMB', 'HML'],
    'CAR' : ['Mkt-Rf', 'SMB', 'HML', 'WML'],
    'FF5' : ['Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA'],
    'HXZ' : ['Mkt-Rf', 'R_ME', 'R_IA', 'R_ROE']
}
sign_dict = {
    'PayoutYield':1,
    'IdioVolAHT':-1,
    'RD':1,
    'AnnouncementReturn':1,
    'VolMkt':-1
    }
# load factors
FF5_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="FF5")
CAR_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="Carhart")
FF5Car_factors = FF5_factors.merge(
    CAR_factors[['month','WML']], on='month', how='left'
)
HXZ_factors = pd.read_excel(data_path + "/Assignment2Data_G5.xlsx", sheet_name="HXZ").drop(columns=['Mkt-Rf'])
factors = FF5Car_factors.merge(
    HXZ_factors,on='month'
)