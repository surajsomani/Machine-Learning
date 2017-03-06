import pandas as pd
import quandl
import math

df = quandl.get("WIKI/GOOGL")
#selecting data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] *100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#label is future price Adj. Close of future
#lets create the variable
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #NA's treatement

forcast_out = int(math.ceil(0.01*len(df)))
#10% of total time in future value to predict
df['label']= df[forecast_col].shift(-forcast_out)
df.dropna(inplace=True)
print(df.head())
print(df.tail())


