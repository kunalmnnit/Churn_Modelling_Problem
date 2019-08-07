# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
a=LabelEncoder()
X[:,1]=a.fit_transform(X[:,1])
X[:,2]=a.fit_transform(X[:,2])

b=OneHotEncoder(categorical_features=[1])
X=b.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.layers import Dense
from keras.models import Sequential

model=Sequential()

model.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
model.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=30,epochs=100)

y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

res = model.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
res = (res > 0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
