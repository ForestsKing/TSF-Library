import argparse
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main

fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Long Time Series Forecasting Library')

# basic
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model', type=str, default='Crossformer',
                    help='model name, options: '
                         '[Reformer, Informer, Autoformer, FEDformer, NLinear, DLinear, TimesNet, Crossformer]')

# data
parser.add_argument('--root_path', type=str, default='./data/ETT-small', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--save_path', type=str, default='./', help='root path of the save file')
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# forecasting
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--percent', type=int, default=100, help='percent of train length')

# basic model
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# DLinear
parser.add_argument('--moving_avg', default=25, help='window size of moving average')

# TimesNet
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

# Informer
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

# Crossformer
parser.add_argument('--seg_len', type=int, default=12, help='segment length (L_seg)')
parser.add_argument('--baseline', action='store_true',
                    help='whether to use mean of past series as baseline for prediction', default=False)
parser.add_argument('--merge_win', type=int, default=2, help='window size for segment merge')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

mse_list, mae_list = [], []
for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        ii
    )

    exp = Exp(args)

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting)
    mse_list.append(mse)
    mae_list.append(mae)

    torch.cuda.empty_cache()
print('>>>>>>>>>>  {}  <<<<<<<<<<'.format(setting[:-2]))
print('MSE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mse_list), np.std(mse_list)))
print('MAE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mae_list), np.std(mae_list)))
print('>>>>>>>>>>  {}  <<<<<<<<<<'.format(setting[:-2]))

f = open("result.txt", 'a')
f.write('>>> {} <<<'.format(setting[:-2]) + "  \n")
f.write('MSE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mse_list), np.std(mse_list)) + "  \n")
f.write('MAE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mae_list), np.std(mae_list)) + "  \n")
f.write('>>> {} <<<'.format(setting[:-2]))
f.write('\n')
f.write('\n')
f.close()
