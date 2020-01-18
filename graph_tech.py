import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()

df = pd.read_csv('employee_reviews.csv')
tech_data = np.genfromtxt(fname = 'employee_reviews.csv',delimiter=',',dtype=float)
labels, tea = pd.factorize(df['company'])
print(len(tech_data))
print(str(tech_data))
print(tech_data.shape)
print(df['company'].value_counts())
#tech_data = df.delete(arr=tech_data,obj=0,axis=1)
objects = ['amazon','apple','facebook','google','microsoft','netflix']
y_pos = np.arange(len(objects))
performance = [3.587363,3.958224,4.511950,4.339430,3.816564,3.411111]
plt.barh(y_pos, performance, align='center', alpha=.5)
plt.yticks(y_pos, objects)
plt.xlabel('Average amount of stars')
plt.title('Overall Ratings Stars per Company')
plt.show()
data = tech_data[:,range(0,6)]
Y = labels
data=data[1:]

print(df.groupby('company')['overall-ratings'].mean())

imp = Imputer(missing_values="NaN",strategy='median',axis=0)
data = imp.fit_transform(data)

dataset = pd.DataFrame({'Column1':data[:,0],'Column2':data[:,1],'Column3':data[:,2],'Column4':data[:,3],'Column5':data[:,4],'Column6':data[:,5],'Column7':labels})
print(dataset)

print(dataset.groupby('Column7')['Column2'].mean())
x="Column"
for i in range(6):
    x=x+str(i+1)
    print(dataset.groupby('Column7')[x].mean())
    x="Column"
