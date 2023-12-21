#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas_datareader


# In[2]:


### Data Collection
import pandas_datareader as pdr
key=""


# In[3]:


df = pdr.get_data_tiingo('ADDYY', api_key='080a93a9c29ff5cdb914031daf88f8a6330e556f')


# In[4]:


df.to_csv('ADDYY.csv')


# In[5]:


import pandas as pd


# In[6]:


df=pd.read_csv('ADDYY.csv')


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df1=df.reset_index()['close']


# In[10]:


df1


# In[11]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[12]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 


# In[13]:


import numpy as np


# In[14]:


df1


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[16]:


print(df1)


# In[17]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[18]:


training_size,test_size


# In[19]:


train_data


# In[20]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[21]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[22]:


print(X_train.shape), print(y_train.shape)


# In[23]:


print(X_test.shape), print(ytest.shape)


# In[24]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[25]:



pip install --upgrade pip


# In[26]:


pip install tensorflow


# In[27]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[28]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[29]:


model.summary()


# In[30]:


model.summary()


# In[31]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[32]:


import tensorflow as tf


# In[33]:


tf.__version__


# In[34]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[35]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[36]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[37]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[38]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[39]:


len(test_data)


# In[40]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[41]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[42]:


temp_input


# In[45]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[46]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[47]:


import matplotlib.pyplot as plt


# In[48]:


len(df1)


# In[49]:


plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[50]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[51]:


df3=scaler.inverse_transform(df3).tolist()


# In[52]:


plt.plot(df3)


# In[ ]:




