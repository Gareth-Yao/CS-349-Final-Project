from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.layers import Dropout
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# Read Data using yahoo finance
data = yf.download(
    tickers=['AMZN'],
    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    period="max",
    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    interval="1d")

num_rolling = 30

#Create the 31 day rolling prices
header = ["day " + str(i) for i in range(1, (num_rolling + 1 + 1))]

closes = data['Close'].values
scaler = MinMaxScaler()
closes_scaled = scaler.fit_transform(closes.reshape(-1, 1))
closes_scaled = closes_scaled.reshape(-1)
df = []
for i in range(num_rolling, len(closes)):
    arr = []
    for j in range(i-num_rolling, i+1):
        arr.append(closes_scaled[j])
    df.append(np.array(arr))
df = pd.DataFrame(df, columns=header)



data_x = []
data_y = []
def saveToLSTMData(x):
    datapoint = []
    for i in range(1, num_rolling + 1):
        timestamp = []
        timestamp.append(x['day ' + str(i)])
        datapoint.append(timestamp)
    data_x.append(datapoint)
    data_y.append(x['day ' + str(num_rolling + 1)])

df.apply(lambda x : saveToLSTMData(x), axis=1)
data_x = np.array(data_x)
data_y = np.array(data_y)
testing_split = 0.2
testing_split = int(len(data_x)*testing_split)
train_x = data_x [:-testing_split]
test_x = data_x[-testing_split:]
train_y = data_y [:-testing_split]
test_y = data_y[-testing_split:]
# indices = np.arange(len(data_x))
# np.random.shuffle(indices)
# data_x = data_x[indices]
# data_y = data_y[indices]

dim = data_x.shape

# Create Model
model = Sequential()
dense_nodes = 5
hidden_nodes = int(len(train_x) / (2 * (dense_nodes  + num_rolling)))
print("Suggested Number of Hidden Node is:", hidden_nodes)
model.add(LSTM(hidden_nodes, return_sequences = False, input_shape = (dim[1], dim[2])))
model.add(Dropout(0.2))
model.add(Dense(dense_nodes))
model.add(Dense(1)) # 1 output: Price


# Train
epochs = 100
train_scores = []
test_scores = []
train_loss = LambdaCallback(on_epoch_end=lambda batch, logs: train_scores.append(logs['loss']))
earlystopper = EarlyStopping(monitor='loss', patience=epochs/10)
model.compile(optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='mae', metrics=[RootMeanSquaredError()])
test_loss = LambdaCallback(on_epoch_end=lambda batch, logs: test_scores.append(model.evaluate(test_x, test_y)[0]))

model.fit(train_x, train_y, batch_size=50, epochs=epochs, callbacks=[train_loss, test_loss, earlystopper])
result = model.evaluate(test_x,test_y)[1]
predictions = model.predict(test_x)
predictions = scaler.inverse_transform(predictions)
test_y = scaler.inverse_transform(test_y.reshape(-1,1))
print(result)

plt.figure()
plt.title("Testing RMSE: " + str(result))
plt.grid()
plt.suptitle("Learning Curve")
plt.ylabel("loss")
plt.xlabel("epochs")
# plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(train_scores),len(train_scores)), train_scores, linewidth=1, color="r",
         label="Training loss")
plt.plot(np.linspace(0,len(test_scores),len(test_scores)), test_scores, linewidth=1, color="b",
          label="Testing loss")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()

plt.clf()

plt.title("Predicted vs Actual")
plt.grid()
plt.ylabel("value")
plt.xlabel("samples")
# plt.ylim(top=max(train_scores),bottom=min(train_scores))
plt.plot(np.linspace(0,len(predictions),len(predictions)), predictions, linewidth=1, color="r",
         label="Predictions")
plt.plot(np.linspace(0,len(test_y),len(test_y)), test_y, linewidth=1, color="b",
          label="Actuals")
legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
legend.get_frame().set_facecolor('C0')

plt.show()