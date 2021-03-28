import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as smf
from sklearn import linear_model
from pandas_datareader import data
import urllib.request
import zipfile


"""
FiveFactorModel class: Gives necessary tools to perform 5 factor model analysis on stocks

Rm-RF : return minus risk free rate
smb : return spread of small minus large stocks (size) -> Small cap companies should outperform the market over the long run 
hml : return spread of cheap minus expensive stock (value) -> High book value to market ratio vs low book to market ratio. High book value to market are value stocks, which generally outperform growth stocks in the long run
rmw : return spread of most profitable minus least profitable -> Most profitable firms do better in the long run
cma : return spread of firms that invest conservatively minus aggressively - Conservative firms do better

Correlation Coefficients:
Alpha - positive alpha is what you are looking for. If the portfolio can be explained by the other factors, there is no added value of the portfolio manager 
SMB - positive correlation means the portfolio is weighted towards small cap companies
HML - positive correlation means that returns are attributable to value premium: That high book value to market ratio stocks generally outperform
RMW - positive correlation means that returns are attributable to profitability: Profitable firms outperform in the long run
CMA - positive correlation means that returns are attributable to conservative investment: Conservative firms outperform in the long run
"""

class FiveFactorModel:
    
    def __init__(self):
        self.FamaFrench = self.getRegressors()
        print(
            """
            FiveFactorModel class: Gives necessary tools to perform 5 factor model analysis on stocks

            Rm-RF : return minus risk free rate
            smb : return spread of small minus large stocks (size) -> Small cap companies should outperform the market over the long run 
            hml : return spread of cheap minus expensive stock (value) -> High book value to market ratio vs low book to market ratio. High book value to market are value stocks, which generally outperform growth stocks in the long run
            rmw : return spread of most profitable minus least profitable -> Most profitable firms do better in the long run
            cma : return spread of firms that invest conservatively minus aggressively - Conservative firms do better

            Correlation Coefficients:
            Alpha - positive alpha is what you are looking for. If the portfolio can be explained by the other factors, there is no added value of the portfolio manager 
            SMB - positive correlation means the portfolio is weighted towards small cap companies
            HML - positive correlation means that returns are attributable to value premium: That high book value to market ratio stocks generally outperform
            RMW - positive correlation means that returns are attributable to profitability: Profitable firms outperform in the long run
            CMA - positive correlation means that returns are attributable to conservative investment: Conservative firms outperform in the long run
            """
        )

    def getRegressors(self):
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        urllib.request.urlretrieve(ff_url,'fama_french.zip')
        zip_file = zipfile.ZipFile('fama_french.zip', 'r')
        zip_file.extractall()
        zip_file.close()
        FamaFrench = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows = 3, index_col = 0)
        FamaFrench = FamaFrench.reset_index().rename(columns={'index': 'Date'})
        FamaFrench = pd.DataFrame(FamaFrench)
        FamaFrench['Date'] = FamaFrench['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

        return FamaFrench 

    def mergeDataFrames(self, assetReturns, FamaFrench):
        df_joined = pd.merge(assetReturns, FamaFrench, left_on=['Date'], right_on = ['Date'])
        df_joined.loc[df_joined.RF == 0, "RF" ] = 0.00001
        df_joined['excess_return'] = df_joined.apply(lambda row: row.rtn - row.RF, axis=1)
        df_joined.fillna(0, inplace=True)
        df_joined.rename(columns={"Mkt-RF":"mkt_excess"}, inplace=True)
        return df_joined
    
    def regression(self, joinedDataFrame):
        X = joinedDataFrame[["mkt_excess", "SMB", "HML", "RMW", "CMA"]]
        Y = joinedDataFrame["excess_return"]
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        X = smf.add_constant(X) # adding a constant
        model = smf.OLS(Y, X).fit()
        #print(model.summary())
        return model

    def fivefactor(self, stock, start_date, end_date):
        assetReturns = data.get_data_yahoo(stock, start=start_date, end=end_date)
        assetReturns = assetReturns.reset_index()
        assetReturns = assetReturns[['Date', 'Adj Close']]
        assetReturns = assetReturns.rename(columns={"Adj Close":"adjust_close"})
        assetReturns['Date'] =  pd.to_datetime(assetReturns['Date'], format='%Y-%m-%d')
        assetReturns['rtn'] = (assetReturns['adjust_close'].pct_change()) 
        joinedDataFrame = self.mergeDataFrames(assetReturns, self.FamaFrench)
        model = self.regression(joinedDataFrame)
        return model



#results = model.params[0]
#results = pd.to_numeric(results)
#p_value = pd.to_numeric(model.summary2().tables[1]['P>|t|'][0])
#return (results, p_value)

        
"""
results = model.params[0]
results = pd.to_numeric(results)
p_values = model.summary2().tables[1]['P>|t|']
p_values = p_values[0]
p_values = pd.to_numeric(p_values)

#print(model.summary())
if results > 0:
    if p_values <= 0.05:
        print('Strong Outperform')
    else:
        print('Perform')
if results < 0:
    if p_values <= 0.05:
        print('Strong Underperform')
    else:
        print('Underperform')
"""
