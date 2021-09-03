# OK nhất từ trước tới giờ
# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from tensorflow.keras.models import Sequential, save_model, load_model

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pd.read_csv('./ETDataset/ETT-small/BTC/Binance_BTCUSDT_Full_7col_date_open_high_low_VoBTC_VoUSDT_close.csv', usecols=[6], engine='python')

cot_ngay = pd.read_csv('./ETDataset/ETT-small/BTC/Binance_BTCUSDT_Full_7col_date_open_high_low_VoBTC_VoUSDT_close.csv', usecols=[0], engine='python').values
cot_ngay = numpy.reshape(cot_ngay,(len(cot_ngay),))

cot_ngay = pd.to_datetime(cot_ngay)

dataset = dataframe.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.9)

look_back = 30

trainX, trainY = create_dataset(train, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))


filepath = './saved_model60'

model = load_model(filepath, compile = True)

# tạo dự đoán
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)





# Đảo ngược scale để hiện về kết quả gốc
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])



# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

df = pd.DataFrame(testPredictPlot)
df.to_csv('file_name.csv', encoding='utf-8')

# plot baseline and predictions
plt.title("Dự đoán giá BTC")
plt.xlabel("Giá")
plt.ylabel("Thời gian")
plt.plot(scaler.inverse_transform(dataset), label='Giá trước dự đoán',color='green')
plt.plot(trainPredictPlot, label='Giá dự đoán train', color='yellow')
plt.plot(testPredictPlot, label='Giá dự đoán test', color='red')
plt.show()
print('kết thúc')

