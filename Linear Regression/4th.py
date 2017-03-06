import pandas as pd
import quandl
import math
#data preprocessing and algo
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np

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
print(forcast_out)
#10% of total time in future value to predict
df['label']= df[forecast_col].shift(-forcast_out)
df.dropna(inplace=True)
#features are everything except label
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
#scale along the training data
x = preprocessing.scale(x)
#print(len(x),len(y))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)
print(accuracy)
#n_jobs how many threads? see in documentation look example 6th_py
