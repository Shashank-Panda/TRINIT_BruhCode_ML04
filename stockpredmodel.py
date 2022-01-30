import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
import matplotlib.pyplot as plt

df = pd.read_csv('daily_IBM.csv')

df.head()

df.tail()

df1 = df.reset_index()['close']
df2 = pd.DataFrame(df1)

print(df1)

scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:1],df1[training_size:len(df1),:1]
train_data,test_data

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

len(X_test), len(y_test)

X_train_1 = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test_1 = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train_1, y_train, validation_data=(X_test, y_test), epochs=1, verbose=1)

train_predict = model.predict(X_train_1)
test_predict = model.predict(X_test_1)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

l = len(df) - len(test_predict)
test = df[l:]
train = df[:l]
test['Predictions'] = test_predict

plt.figure(figsize=(22, 10))
plt.plot(test[['Predictions', 'close']])
plt.plot(train.close)
plt.legend(['Predicted', 'Actual', 'Actual-Train'])

x_input=df1[5498:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<50):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

future_p = [i[0] for i in scaler.inverse_transform(lst_output)]
print(future_p)

future_pred = pd.DataFrame(df1)
for i in range(50):
  future_pred.loc[future_pred.shape[0]] = [None]
print(future_pred)
abcd = future_pred[(len(df2)):]

abcd['predictions'] = future_p

plt.figure(figsize=(22, 10))
plt.plot(test[['Predictions', 'close']])
plt.plot(train.close)
plt.plot(abcd['predictions'])
plt.legend(['Predicted', 'Actual', 'Actual-Train', 'Future'])

def predict_stock(input):
    scaled_input = scaler.transform(input)
    scaled_input , _ = create_dataset(scaled_input, time_step)
    scaled_input = scaled_input.reshape(scaled_input.shape[0],scaled_input.shape[1] , 1)
    scaled_output = model.predict(scaled_input)
    output = scaler.inverse_transform(scaled_output)
    return output

print(predict_stock(scaler.inverse_transform(test_data)))