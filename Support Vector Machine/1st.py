import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
#df.dropna(inplace=True)
#remove the id column
df.drop(['id'],1,inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train,Y_train)

accuracy = clf.score(X_test,Y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(2,-1) #2 sample so (2,-1)
#for unknown number we write (len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)
