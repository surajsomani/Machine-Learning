import pandas as pd
import quandl, math, datetime
#data preprocessing and algo
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import pickle


style.use('ggplot')


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
#print(forcast_out)

#10% of total time in future value to predict
df['label']= df[forecast_col].shift(-forcast_out)
#features are everything except label
x = np.array(df.drop(['label'],1))
#scale along the training data
x = preprocessing.scale(x)

x_lately = x[-forcast_out:]
x = x[:-forcast_out:] 
df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x,y,test_size=0.2)
#model training
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,Y_train)


#saving the classifier
with open('linearregression.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test,Y_test)
#model prediction
forcast_set = clf.predict(x_lately)
print(forcast_set,accuracy,forcast_out)

#plotting

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print(df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
