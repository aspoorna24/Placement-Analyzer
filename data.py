import pandas as pd
import numpy as np
import pickle

d = pd.read_csv('Placement_Data.csv')
d = pd.get_dummies(d,columns=['Placed',],drop_first = True)

Y = d[['Placed_Yes']]
X = d[['Branch','SSLC','PUC','BE','Arrears','Internships','C','C++','Java','Python','JavaScript','PHP','Data_Structure','DBMS','Web_dev','App_dev','PCB',]]

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.1,random_state=1)

from sklearn.linear_model import LogisticRegression
x = LogisticRegression().fit(Xtrain,Ytrain)


pickle.dump(x, open('data.pkl','wb'))