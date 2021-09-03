import numpy as np
from sklearn.preprocessing.data import scale
import pandas as pd
# importing package
import matplotlib.pyplot as plt
import numpy as np





  
# create data
x = [1,2,3,4,5]
y = [3,3,3,3,3]
  
# plot lines
plt.plot(x, y, label = "line 1")
plt.plot(y, x, label = "line 2")
plt.plot(x, np.sin(x), label = "curve 1")
plt.plot(x, np.cos(x), label = "curve 2")
plt.legend()
plt.show()


























ngay = pd.date_range(start='2021-08-22T19:00:00.000000000', periods=2, freq='h')

print(ngay)



arr = np.array(
  [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9]
  ]
)

# ok
def SlidingWindow (arr, element):
  arr = np.append(arr, element)
  new_a = np.delete(arr, 0)
  return arr


#chuẩn bị dự báo 


print('xong')


















arr = np.array(
  [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9]
  ]
)

arr = arr.reshape(3,3,1)

#print(arr)
#print(arr.shape)
#print(arr.shape[0])


from sklearn.preprocessing import MinMaxScaler

b = np.array(
  [
    [1],
    [2]
  ]
)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(b)
b= scaler.transform(b)
print('mảng B\n',b)


d = np.array(
  [
    [2],
    [4]
  ]
)
d = scaler.transform(d)
print('mảng D\n',d)

print('đảo ngược B\n',scaler.inverse_transform(b))
print('đảo ngược D\n',scaler.inverse_transform(d))


0.02264102
0.02278699
0.02319245
0.02337085
0.02473321

