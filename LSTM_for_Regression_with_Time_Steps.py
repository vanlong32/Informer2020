# OK nhất từ trước tới giờ
# link bài viết: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# LSTM for international airline passengers problem with time step regression framing
from json import load
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

# split into train and test sets
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3

# THH xác định chỗ này có phải trả về 3 bước thời gian trước đó
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# THH reshape có vẻ như reshape (arr, (hình dạng))
# chuyển mảng thành 3 chiều. 
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
filepath = './saved_modelv2'

model = load_model(filepath, compile = True)

#model = Sequential()

model.add(LSTM(4, input_shape=(look_back, 1),return_sequences=False))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

#save_model(model, filepath)






# ============================================

# dự đoán
dataset = pd.read_csv('./ETDataset/ETT-small/BTC/BTC_dubao.csv', usecols=[6], engine='python').values
dataset = dataset.astype('float32')
scaler2 = MinMaxScaler(feature_range=(0, 1))
dataset = scaler2.fit_transform(dataset)

def SlidingWindow (arr, element):
  arr = numpy.append(arr, element)
  arr = numpy.delete(arr, 0)
  return arr

ketqua = dataset;


# Dự đoán vòng đầu
data_dudoan = numpy.reshape(dataset, (1,look_back,1))

dudoan = model.predict(data_dudoan)
ketqua = numpy.append(ketqua, dudoan)

data_dudoan = SlidingWindow(data_dudoan,dudoan)

for x in range(24):
	data_dudoan = numpy.reshape(data_dudoan, (1,look_back,1))
	dudoan = model.predict(data_dudoan)
	ketqua = numpy.append(ketqua, dudoan)
	data_dudoan = SlidingWindow(data_dudoan,dudoan)


ketqua = numpy.reshape(ketqua, (ketqua.shape[0],1))
ketqua = scaler2.inverse_transform(ketqua)
df = pd.DataFrame(ketqua)
df.to_csv('./ETDataset/ETT-small/BTC/ket_qua_du_doan.csv', encoding='utf-8')


# ============================================









# tạo dự đoán
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


testPredict = numpy.reshape(testPredict, (testPredict.shape[0],1))

# Đảo ngược scale để hiện về kết quả gốc
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])




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


# plot baseline and predictions
plt.title("Dự đoán giá BTC")
plt.xlabel("Giá")
plt.ylabel("Thời gian")
plt.plot(scaler.inverse_transform(dataset), label='Giá trước dự đoán',color='green')
plt.plot(trainPredictPlot, label='Giá dự đoán train', color='yellow')
plt.plot(testPredictPlot, label='Giá dự đoán test', color='red')
plt.show()
print('kết thúc')















a = numpy.array(
	[
		[
			[0.02264102],
			[0.02278699],
			[0.02319245]
		],
		[
			[0.02278699],
			[0.02319245],
			[0.02010408]
		]	
		
	]

)
# 0.01671556
du_doan = model.predict(a)
print('xong')







# tạo mảng 3 chiều
#b,c = create_dataset(a,3)
#b = numpy.reshape(b, (b.shape[0], b.shape[1], 1))

#du_doan = model.predict(b)[0,0]

#a = numpy.append(a, [[du_doan]]) 






# tạo vòng lặp predic
#for i in range(6):
#  a[0]
#
#  print(i)



a_inver = scaler.inverse_transform(a)

plt.title("Dự đoán giá BTC")
plt.xlabel("Giá")
plt.ylabel("Thời gian")
plt.plot(a_inver, label='Giá trước dự đoán')
plt.show()
print('kết thúc')