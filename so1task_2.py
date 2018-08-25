from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tornado

perUserDict = {}
#data = pd.read_csv('~/Research/SO1/Fathi/train.csv')
data_path = ['.','.', 'data']
filepath = os.sep.join(data_path + ['train.csv'])
data = pd.read_csv(filepath)


# In[6]:


numCustomers = data.i.nunique() #starts at zero to numCustomers -1
numProducts = data.j.nunique() # starts at zero to numProducts -1
numSteps = data.t.nunique() # starts at zero to numSteps -1
print(numCustomers, numProducts , numSteps)


# In[7]:


X = np.zeros((numSteps,numProducts)) # Every Cell Value is whether this product was advertised or not
Y = np.zeros((numSteps,numCustomers,numProducts)) #time , customer , product and the value in every Cell
#is whether the product was bought or not.


# In[8]:


print(X.shape)
print(Y.shape)


# In[9]:


for index, row in data.iterrows():
    time = int(row['t'])
    customer = int(row['i'])
    product = int(row['j'])
    advertised = int(row['advertised'])
    Y[time,customer, product] = 1
    X[time,product] = advertised


# In[10]:


WindowSize = 20 
PredictionStep = 40


# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.layers import LSTM


# In[15]:


print(keras.__version__)


# In[16]:


numHiddenNeurons = 150
model = Sequential()
model.add(LSTM(numHiddenNeurons , input_shape=(None,numProducts), return_sequences=True))
model.add(Dense(numHiddenNeurons))
model.add(Dropout(.5))
#model.add(Dense((numCustomers * numProducts),activation='sigmoid'))
model.add(Dense(numCustomers * numProducts))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())


# In[ ]:


tempX = np.zeros((WindowSize, X.shape[1]))
tempY = np.zeros((WindowSize, Y.shape[1], Y.shape[2]))
numEpochs = 100
batchSize = 2
NumTimeSteps = WindowSize//batchSize
print("Time step is ", NumTimeSteps)
for j in range(numEpochs):
    print("Now In Epoch :: " , j)
    for i in range(0, PredictionStep, 1):
      tempX = X[i:i+WindowSize,:]
      tempY = Y[i:i+WindowSize,:]
#     print(tempX.shape)
      if(tempX.shape[0] >= WindowSize):
          tempX = tempX.reshape(batchSize, NumTimeSteps,numProducts)
          tempY = tempY.reshape(batchSize,NumTimeSteps,numCustomers * numProducts)
      else:
          tempX = tempX.reshape(1,tempX.shape[0], numProducts)
          tempY = tempY.reshape(1, tempY.shape[0], numCustomers* numProducts)
      model.fit(tempX, tempY, batch_size=batchSize, epochs=1)


# In[17]:


model.save("200_Neurons_w_Dropout_255_epochs_final")


# In[ ]:


model.load_weights("200_Neurons_w_Dropout_255_epochs_final")


# In[ ]:


TestWindowSize = 25 # real windowSize while testing.
print(X.shape)
for i in range(0, numSteps - TestWindowSize+1, 1):
    print(i)
    tempX = X[i:i+TestWindowSize,:]
    tempY = Y[i:i+TestWindowSize,:]
    tempX = tempX.reshape(1,TestWindowSize,numProducts)
    tempY = tempY.reshape(1,TestWindowSize,numCustomers * numProducts)
    print(model.evaluate(tempX,tempY))
    


# In[ ]:


# to predict the value at t = desiredTime, use the previous (window_size) steps before that
desiredTime = 37
TestWindowSize = 25
Y_pred = model.predict(X[desiredTime-TestWindowSize +1:desiredTime+1, :].reshape(1, TestWindowSize, numProducts))
Y_pred = Y_pred[0,-1,:].reshape(numCustomers , numProducts) #last element
print("YPred",Y_pred.shape)
Y_true = Y[desiredTime].reshape(numCustomers, numProducts)
print("Y_ytrue",Y_true.shape)
Y_pred[Y_pred >= .5] = 1
Y_pred[Y_pred < .5] = 0


# In[ ]:


mse = ((Y_true - Y_pred) ** 2).mean()
print(mse)

counterShared = 0
counterPredOnly = 0
counterTrueOnly = 0
for i in range(numCustomers):
    for j in range(numProducts):
        if(Y_pred[i][j] == 1 and Y_true[i][j] ==1):
            #print(i,j)
            counterShared = counterShared +1
        elif(Y_pred[i][j] == 1):
            counterPredOnly = counterPredOnly +1
        elif(Y_true[i][j] ==1):
            counterTrueOnly = counterTrueOnly +1
            
            
print( "Shared buying between true and prediction : ",counterShared)
print("Elements really bought that weren't predicted to be bought", counterTrueOnly)
print("Elements That were falsly predicted to be bought", counterPredOnly)


# In[ ]:


#Evaluating at week = 50

data_50 = pd.read_csv('promotion_schedule.csv')

#note in the file itself, the data for product j = 20 is missing. I will add it in code here
data_50 = data_50.append(pd.DataFrame({'j':20 , 'discount':0.0, 'advertised':0},index=[39])).sort_values(by=['j'])
data_50 = data_50.reset_index(drop=True)

X_50 = data_50.advertised.values.reshape(1,numProducts)
print(X_50.shape)

X_for_Evaluation = np.copy(X)
X_for_Evaluation = np.vstack((X_for_Evaluation,X_50))
print(X_for_Evaluation.shape)


# In[ ]:


TestWindowSize = 25
desiredTime = 49
Y_pred = model.predict(X_for_Evaluation[desiredTime-TestWindowSize +1:desiredTime+1, :].reshape(1, TestWindowSize, numProducts))
Y_pred = Y_pred[0,-1,:].reshape(numCustomers , numProducts) #last element
#print("YPred",Y_pred.shape)

Y_pred[Y_pred < .00000000001] = 0

with open('Predictions.csv', 'w', newline='') as csvfile:
    Writer = csv.writer(csvfile)
    Writer.writerow(["i","j","prediction"])
    for i in range(numCustomers):
        for j in range (numProducts):
            print(i,",",j, ",", Y_pred[i][j])
            Writer.writerow([i,j,Y_pred[i][j]])
