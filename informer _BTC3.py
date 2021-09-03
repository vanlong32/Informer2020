'''
    Link gốc dự án: https://github.com/zhouhaoyi/Informer2020

'''

import os
import sys
if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']

import numpy as np
import matplotlib.pyplot as plt

from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch

# !pip install -r ./Informer2020/requirements.txt

"""## Experiments: Train and Test"""



def init(): 
    
    global setting
    global exp
    global args

    args = dotdict()

    args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = 'custom' # data
    args.root_path = './ETDataset/ETT-small/BTCscale/Train2/' # root path of data file
    args.data_path = 'Binance_BTCUSDT_Full_5col.csv' # data file
   

    args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = 'close' # target feature in S or MS task
    args.freq = 'd' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.checkpoints = './informer_checkpoints' # location of model checkpoints

    args.seq_len = 96 # input sequence length of Informer encoder
    args.label_len = 48 # start token length of Informer decoder
    args.pred_len = 24 # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 4 # encoder input size
    args.dec_in = 4 # decoder input size
    args.c_out = 1 # output size
    args.factor = 4 # probsparse attn factor
    args.d_model = 1024 # dimension of model
    args.n_heads = 5 # num of heads
    args.e_layers = 2 # num of encoder layers
    args.d_layers = 1 # num of decoder layers
    args.d_ff = 2048 # dimension of fcn in model
    args.dropout = 0.05 # dropout
    args.attn = 'prob' # attention used in encoder, options:[prob, full]
    args.embed = 'timeF' # tìm hiểu thêm chỗ này time features encoding, options:[timeF, fixed, learned]
    args.activation = 'sigmoid' # activation sigmoid relu
    args.distil = True # whether to use distilling in encoder
    args.output_attention = False # whether to output attention in ecoder
    args.mix = True
    args.padding = 0
    args.detail_freq = 'd'

    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 1
    args.patience = 3
    args.des = 'exp'

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # set experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, 0)
    
    Exp = Exp_Informer

    exp = Exp(args)
    

    return args


def Train():
    
    for ii in range(args.itr):
        # setting record of experiments
        
        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        # test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()


def Predict ():

    exp.predict(setting, True)

    prediction = np.load('./results/'+setting+'/real_prediction.npy')

    prediction.shape



    preds = np.load('./results/'+setting+'/pred.npy')
    trues = np.load('./results/'+setting+'/true.npy')

    # [samples, pred_len, dimensions]
    preds.shape, trues.shape

    # draw OT prediction
    plt.figure()
    plt.title("Load kết quả test")
    plt.plot(trues[0,:,-1], label='Giá trị thực')
    plt.plot(preds[0,:,-1], label='Dự đoán')
    plt.legend()
    plt.show()

def main(): 
    
    # Chuẩn bị cấu hình
    init()

    # huấn luyện
    #Train()

    args.root_path = './ETDataset/ETT-small/BTCscale/Train2/' # root path of data file

    #args.data_path = 'Binance_BTCUSDT_Full_5col_date_open_high_low_close_VungTrain.csv' # data file    
    args.data_path = 'test.csv'
    
    Predict ()



if __name__ == "__main__":
    main()























# import numpy as np

# """## Visualization"""

# # When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# # The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

# # Khi chúng tôi hoàn thành exp.train (cài đặt) và exp.test (cài đặt), 
# # chúng tôi sẽ nhận được một mô hình được đào tạo và kết quả của thử nghiệm kiểm tra
# # Kết quả của thử nghiệm kiểm tra sẽ được lưu trong ./results/{setting}/pred. npy 
# # (dự đoán của tập dữ liệu thử nghiệm) và ./results/{setting}/true.npy 
# # (cơ sở của tập dữ liệu thử nghiệm)

# preds = np.load('./results/'+setting+'/pred.npy')
# trues = np.load('./results/'+setting+'/true.npy')

# # [samples, pred_len, dimensions]
# preds.shape, trues.shape

# print(preds.shape)
# print(trues.shape)

# import matplotlib.pyplot as plt
# import seaborn as sns



# a = trues[0,:,-1]
# b = preds[0,:,-1]

# # draw OT prediction
# plt.figure()
# plt.plot(trues[0,:,-1], label='GroundTruth')
# plt.plot(preds[0,:,-1], label='Prediction')
# plt.legend()
# plt.show()

# # draw HUFL prediction
# plt.figure()
# plt.plot(trues[0,:,0], label='GroundTruth')
# plt.plot(preds[0,:,0], label='Prediction')
# plt.legend()
# plt.show()

# from data.data_loader import Dataset_ETT_hour
# from torch.utils.data import DataLoader

# Data = Dataset_ETT_hour
# timeenc = 0 if args.embed!='timeF' else 1
# flag = 'test'; shuffle_flag = False; drop_last = True; batch_size = 1

# data_set = Data(
#     root_path=args.root_path,
#     data_path=args.data_path,
#     flag=flag,
#     size=[args.seq_len, args.label_len, args.pred_len],
#     features=args.features,
#     timeenc=timeenc,
#     freq=args.freq
# )
# data_loader = DataLoader(
#     data_set,
#     batch_size=batch_size,
#     shuffle=shuffle_flag,
#     num_workers=args.num_workers,
#     drop_last=drop_last)

# import os

# args.output_attention = True

# exp = Exp(args)

# model = exp.model

# setting = 'informer_custom_ftM_sl96_ll48_pl30_dm1024_nh5_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
# path = os.path.join(args.checkpoints,setting,'checkpoint.pth')
# model.load_state_dict(torch.load(path))

# # attention visualization
# idx = 0
# for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(data_loader):
#     if i!=idx:
#         continue
#     batch_x = batch_x.float().to(exp.device)
#     batch_y = batch_y.float()

#     batch_x_mark = batch_x_mark.float().to(exp.device)
#     batch_y_mark = batch_y_mark.float().to(exp.device)
    
#     dec_inp = torch.zeros_like(batch_y[:,-args.pred_len:,:]).float()
#     dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float().to(exp.device)
    
#     outputs,attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

# attn[0].shape, attn[1].shape #, attn[2].shape

# layer = 0
# distil = 'Distil' if args.distil else 'NoDistil'
# for h in range(0,8):
#     plt.figure(figsize=[10,8])
#     plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
#     A = attn[layer][0,h].detach().cpu().numpy()
#     ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
#     plt.show()

# layer = 1
# distil = 'Distil' if args.distil else 'NoDistil'
# for h in range(0,8):
#     plt.figure(figsize=[10,8])
#     plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
#     A = attn[layer][0,h].detach().cpu().numpy()
#     ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
#     plt.show()

# """## Custom Data

# Custom data (xxx.csv) has to include at least 2 features: `date`(format: `YYYY-MM-DD hh:mm:ss`) and `target feature`.

# """

# from data.data_loader import Dataset_Custom
# from torch.utils.data import DataLoader
# import pandas as pd
# import os

# # custom data: xxx.csv
# # data features: ['date', ...(other features), target feature]

# # we take ETTh2 as an example
# args.root_path = './ETDataset/ETT-small/'
# args.data_path = 'ETTh2.csv'

# df = pd.read_csv(os.path.join(args.root_path, args.data_path))

# df.head()

# '''
# We set 'HULL' as target instead of 'OT'

# The following frequencies are supported:
#         Y   - yearly
#             alias: A
#         M   - monthly
#         W   - weekly
#         D   - daily
#         B   - business days
#         H   - hourly
#         T   - minutely
#             alias: min
#         S   - secondly
# '''

# args.target = 'HULL'
# args.freq = 'h'

# Data = Dataset_Custom
# timeenc = 0 if args.embed!='timeF' else 1
# flag = 'test'; shuffle_flag = False; drop_last = True; batch_size = 1

# data_set = Data(
#     root_path=args.root_path,
#     data_path=args.data_path,
#     flag=flag,
#     size=[args.seq_len, args.label_len, args.pred_len],
#     features=args.features,
#     timeenc=timeenc,
#     target=args.target, # HULL here
#     freq=args.freq # 'h': hourly, 't':minutely
# )
# data_loader = DataLoader(
#     data_set,
#     batch_size=batch_size,
#     shuffle=shuffle_flag,
#     num_workers=args.num_workers,
#     drop_last=drop_last)

# batch_x,batch_y,batch_x_mark,batch_y_mark = data_set[0]

