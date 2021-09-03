from utils.tools import StandardScaler


import pandas as pd



from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

import numpy as np





text = '''\
cot1 cot2 cot3
1 2 3
4 5 6
'''

#data1 = pd.read_csv(StringIO(text), sep=' ', header=0)

#print(data1 )






def loaddata ():
    df_raw = pd.read_csv('./ETDataset/ETT-small/Binance_BTCUSDT_4col_date_open_close_TT.csv')
    
    return df_raw

def scale ():
    scaler = StandardScaler()

    scaler.fit(train_data.values)

    caler.transform(df_data.values)

    scaler.inverse_transform(data)



df_raw = loaddata ()

#print(df_data)

seq_len = 24*4*4
label_len = 24*4
pred_len = 24*4

# lấy lengt() của dataset chia tỉ lệ 70% 20% 
num_train = int(len(df_raw)*0.94)
num_test = int(len(df_raw)*0.03)
# vali nghĩa là 
num_vali = len(df_raw) - num_train - num_test

border1s = [0, num_train-seq_len, len(df_raw)-num_test-seq_len]
border2s = [num_train, num_train+num_vali, len(df_raw)]
border1 = border1s[0]
border2 = border2s[0]


# Cắt lấy dòng tiêu đề và loại bỏ cột date. Dữ liệu bên trong: Index(['open', 'close', 'TT'], dtype='object')
cols_data = df_raw.columns[1:]

# loại bỏ cột date
df_data = df_raw[cols_data]


train_data = df_data[border1s[0]:border2s[0]]


scaler = StandardScaler()

df_data_values = train_data.values

c = df_data_values[0]


#a = scaler.fit(train_data.values)

data = scaler.transform(df_data.values)

inver_data = scaler.inverse_transform(data)

# KẾT QUẢ đảo ngược đúng
b = inver_data[0]

df_stamp = df_raw[['date']][border1:border2]           





def DuBao():
    setting = 'informer_custom_ftM_sl96_ll48_pl24_dm1024_nh3_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'

    preds = np.load('./results/'+setting+'/pred.npy')
    trues = np.load('./results/'+setting+'/true.npy')   
    
    
    preds.shape, trues.shape
    
    preds_inver = scaler.inverse_transform(preds)
    trues_inver = scaler.inverse_transform(trues)
    
    print (preds_inver)
    print(trues_inver)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # draw OT prediction
    plt.figure()
    plt.plot(trues[0,:,-1], label='GroundTruth')
    plt.plot(preds[0,:,-1], label='Prediction')
    plt.legend()
    plt.show()

DuBao()

print('xong')

