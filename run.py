# Module sys trong Python cung cấp các hàm và các biến được sử dụng để thao tác các phần khác nhau của môi trường chạy Python.
# Nó cho phép chúng ta truy cập các tham số và chức năng cụ thể của hệ thống.
import sys

# nếu không tìm thấy thư mục Informer2020 thì sửa lại đường dẫn
if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']

# nhập từ tư viện informer2000, không biết để làm gì 
from utils.tools import dotdict

# nhập từ tư viện informer2000, không biết để làm gì 
from exp.exp_informer import Exp_Informer

# nhập thư viện PyTorch là một framework được xây dựng dựa trên python cung cấp nền tảng 
# tính toán khoa học phục vụ lĩnh vực Deep learning
import torch


args = dotdict()

args.model = 'informer' # model của thử nghiệm, tùy chọn: [informer, informerstack, informerlight(TBD)]

args.data = 'ETTh2' # data
args.root_path = './ETDataset/ETT-small/' # root path of data file

# ETT là bộ dữ liệu của nhà máy điện
# ETTm1.csv, ETTm2.csv là Bộ dữ liệu ghi theo phút  (ký hiệu m)
# ETTh1.csv, ETTh2.csv là Bộ dữ liệu ghi theo giờ (ký hiệu bằng h)
args.data_path = 'ETTh2.csv' 


# forecasting task, tùy chọn:[M, S, MS]; 
#   - M:multivariate predict multivariate (đa biến dự đoán đa biến)
#   - S:univariate predict univariate (đơn biến dự đoán đơn biến)
#   - MS:multivariate predict univariate (dự đoán đa biến đơn biến)

args.features = 'M' 

# lựa chọn mục tiêu trong chế độ S or MS task
args.target = 'OT' 

# freq for time features encoding (tần số cho nhiệm vụ dự đoán)
#   tùy chọn :[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], 
#   bạn cũng có thể sử dụng freq chi tiết hơn như 15 phút hoặc 3 giờ

args.freq = 'h' 

# Cài vị trí lưu model checkpoints
args.checkpoints = './informer_checkpoints' 

# input sequence length of Informer encoder (độ dài chuỗi đầu vào của bộ mã hóa Informer)
args.seq_len = 96 

# start token length of Informer decoder (độ dài mã thông báo bắt đầu của bộ giải mã Inform)
args.label_len = 48 

# prediction sequence length (độ dài chuỗi dự đoán)
args.pred_len = 24



# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 7 # encoder input size
args.dec_in = 7 # decoder input size
args.c_out = 7 # output size
args.factor = 5 # probsparse attn factor (hệ số suy giảm xác suất)
args.d_model = 512 # dimension of model (kích thước của mô hình)
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention (chú ý) used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding (tính năng mã hóa thời gian), options:[timeF, fixed, learned]
args.activation = 'gelu' # activation (activation)
args.distil = True # whether to use distilling in encoder (có nên sử dụng phương pháp chưng cất trong bộ mã hóa không)
args.output_attention = False # whether to output attention in ecoder (có tạo sự chú ý trong bộ sinh thái không)
args.mix = True
args.padding = 0
args.freq = 'h'

args.batch_size = 32 
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 6
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

# Set augments by using data name
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}

# Date,Open,High,Low,Close,Adj Close,Volume
# data: nhập tên data
# T: là tên cột lựa chọn làm mục tiêu dự đoán
# M
#data_parser = {
#     'BTC-USD':{'data':'BTC-USD.csv','T':'Close','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]}
#}
    


if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]



print(args.data)


args.detail_freq = args.freq
args.freq = args.freq[-1:]


print('Args in experiment:')
print(args)

Exp = Exp_Informer


for ii in range(args.itr):
    # setting record của thí nghiệm
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

    # set thí nghiệm
    exp = Exp(args)
    
    # train
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    # test
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()



import os

# Đường dẫn lưu mô hình
setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
# path = os.path.join(args.checkpoints,setting,'checkpoint.pth')

# If you already have a trained model, you can set the arguments and model path, then initialize a Experiment and use it to predict
# Prediction is a sequence which is adjacent to the last date of the data, and does not exist in the data
# If you want to get more information about prediction, you can refer to code `exp/exp_informer.py function predict()` and `data/data_loader.py class Dataset_Pred`

exp = Exp(args)

exp.predict(setting, True)


# the prediction will be saved in ./results/{setting}/real_prediction.npy
import numpy as np

prediction = np.load('./results/'+setting+'/real_prediction.npy')

prediction.shape

# Đây là code chi tiết của function predict

def predict(exp, setting, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')
        
    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

    exp.model.eval()
        
    preds = []
        
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        # decoder input
        if exp.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        elif exp.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder
        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if exp.args.features=='MS' else 0
        batch_y = batch_y[:,-exp.args.pred_len:,f_dim:].to(exp.device)
        
        pred = outputs.detach().cpu().numpy()#.squeeze()
        
        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    
    # result save
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    np.save(folder_path+'real_prediction.npy', preds)
    
    return preds
# bạn có thể sử dụng prediction function để lấy kết quả dự đoán
prediction = predict(exp, setting, True)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(prediction[0,:,-1])
plt.show()



