import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import pandas as pd
from pylab import *

df = pd.read_csv('tech_data.csv')
ree=[]
t=arange(0.0,100.0,1)

tech_data = np.genfromtxt(fname = 'tech_data.csv',delimiter=',',dtype=float)
labels, tea = pd.factorize(df['company'])
print(tech_data)
print(len(tech_data))
print(str(tech_data))
print(tech_data.shape)

#tech_data = df.delete(arr=tech_data,obj=0,axis=1)

X = tech_data[:,range(2,9)]
Y = labels
X=X[1:]

imp = Imputer(missing_values="NaN",strategy='median',axis=0)
X = imp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size =.1, random_state=100)
y_train = y_train.ravel()
y_test = y_test.ravel()

for K in range(100):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform',algorithm='auto')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    ree.append(accuracy_score(y_test,y_pred)*100)
    print("accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
title("KNN Accuracy")
xlabel("Iterration")
ylabel("% accurate")
plot(t,ree)
show()
               
