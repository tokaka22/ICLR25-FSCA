import sys
sys.path.append('.')

# import os
# os.environ['TRANSFORMERS_CACHE'] = '/code/huggingface/hub'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import os
import torch
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import random
import numpy as np
import wandb

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# patching
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)

# normal
parser.add_argument('--sweep_flag', type=int, default=0) # auto deploy data
parser.add_argument('--test_flag', type=int, default=0) 
parser.add_argument('--wandb_flag', type=int, default=1) # debug close wandb
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wd_project', default='time_llm_shortterm_test', type=str)

# model
parser.add_argument('--split_len', type=int, default=2)
parser.add_argument('--layer_index', type=str, help='A space-separated string of integers')
parser.add_argument('--w_l2s_flag', type=int, default=0) #TODO learnable
parser.add_argument('--w_l2s_v', type=float, default=0.0001) 

parser.add_argument('--d_l_comp', type=int, default=336)
parser.add_argument('--in_dropout', type=float, default=0)
parser.add_argument('--out_dropout', type=float, default=0.1)

parser.add_argument('--patch_size_stride', type=str, help='A space-separated string of integers')
parser.add_argument('--offline', type=int, default=0)

args = parser.parse_args()

if args.offline == 1:
    os.environ["WANDB_MODE"] = "offline"

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parts = args.layer_index.split('*')
args.gpt_layers = int(parts[0]) if parts[0] else None

if len(parts) > 1 and parts[1]:
    args.gnn_layer_index = [int(x) for x in parts[1].split('_')]
    args.gnn_layer_index_str = '_'.join(str(x) for x in args.gnn_layer_index)
else:
    args.gnn_layer_index = []
    args.gnn_layer_index_str = ""

if len(parts) > 2 and parts[2]:
    args.l_gnn_layer_index = [int(x) for x in parts[2].split('_')]
    args.l_gnn_layer_index_str = '_'.join(str(x) for x in args.l_gnn_layer_index)
else:
    args.l_gnn_layer_index = []
    args.l_gnn_layer_index_str = ""

args.patch_size, args.stride = [int(x) for x in args.patch_size_stride.split()]

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
elif args.task_name == 'classification':
    Exp = Exp_Classification
else:
    Exp = Exp_Long_Term_Forecast

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        group_name = '{}_{}_sl{}_pl{}_df{}_gl{}_{}*{}_s{}_wf{}_wv{}_l{}_b{}_s{}_dl{}_dm{}_id{}_od{}_p{}_s{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_ff,
            args.gpt_layers,
            args.gnn_layer_index_str,
            args.l_gnn_layer_index_str,
            args.split_len,
            args.w_l2s_flag,
            args.w_l2s_v,
            args.learning_rate,
            args.batch_size,
            args.seed,
            args.d_l_comp,
            args.d_model,
            args.in_dropout,
            args.out_dropout,
            args.patch_size,
            args.stride
            )

        setting = '{}'.format(
            args.seasonal_patterns,)

        import datetime
        import uuid
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_string = uuid.uuid4().hex[:6]

        args.it_fname = os.path.join(args.wd_project, group_name + '_' + setting, f'seed-{args.seed}' + f'-itr-{ii}', f'{current_time}_{random_string}')

        wandb_group_name = f'{group_name}'

        wandb_run_name = f'{setting}_seed{args.seed}_it{ii}'

        ### wandb
        if args.wandb_flag:
            run = wandb.init(
                # Set the project where this run will be logged
                project=f"{args.wd_project}", 
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=wandb_run_name,
                group=wandb_group_name,
                config=args) 
        else:
            run = wandb.init(mode="disabled")

        exp = Exp(args)  # set experiments

        print('>>>>>>>start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()

        print('>>>>>>>test_metric only this group : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test_metric(setting, all=0)
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting, test=1)
    exp.test_metric(setting, all=1)
    torch.cuda.empty_cache()
