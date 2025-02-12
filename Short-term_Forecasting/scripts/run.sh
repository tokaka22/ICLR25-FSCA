. activate FSCA
cd /code/TimeLLM/code/ICLR25-FSCA-official/Short-term_Forecasting

# year
CUDA_VISIBLE_DEVICES=0 python run.py --batch_size=16 --c_out=1 --d_ff=128 --d_l_comp=128 --d_model=768 --data=m4 --dec_in=1 --des=Exp --enc_in=1 --features=M --is_training=1 --itr=1 --layer_index=4*0_3_4*0_3_4 --learning_rate=0.005 --loss=SMAPE --model=FSCA --patience=20 --root_path=./datasets/m4 --seasonal_patterns=Yearly --seed=42 --split_len=2 --patch_size_stride="1 1" --task_name=short_term_forecast --test_flag=0 --train_epochs=100 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1 --wd_project=fsca-st-log00

# quart
CUDA_VISIBLE_DEVICES=0 python run.py --batch_size=16 --c_out=1 --d_ff=336 --d_l_comp=256 --d_model=512 --data=m4 --dec_in=1 --des=Exp --enc_in=1 --features=M --in_dropout=0 --is_training=1 --itr=1 --layer_index=4*0*0 --learning_rate=0.0025 --loss=SMAPE --model=FSCA --out_dropout=0.1 --patch_size_stride="4 2" --patience=5 --root_path=./datasets/m4 --seasonal_patterns=Quarterly --seed=42 --split_len=2 --task_name=short_term_forecast --test_flag=0 --train_epochs=100 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1 --wd_project=fsca-st-log00

# month
CUDA_VISIBLE_DEVICES=0 python run.py --batch_size=16 --c_out=1 --d_ff=336 --d_l_comp=768 --d_model=336 --data=m4 --dec_in=1 --des=Exp --enc_in=1 --features=M --in_dropout=0 --is_training=1 --itr=1 --layer_index=4*0*0 --learning_rate=0.004 --loss=SMAPE --model=FSCA --out_dropout=0 --patch_size_stride="4 2" --patience=5 --root_path=./datasets/m4 --seasonal_patterns=Monthly --seed=42 --split_len=2 --task_name=short_term_forecast --test_flag=0 --train_epochs=100 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1 --wd_project=fsca-st-log00

# week
CUDA_VISIBLE_DEVICES=0 python run.py --batch_size=16 --c_out=1 --d_ff=512 --d_model=768 --data=m4 --dec_in=1 --des=Exp --enc_in=1 --features=M --in_dropout=0 --is_training=1 --itr=1 --layer_index=4*0*0 --learning_rate=0.001 --loss=SMAPE --model=FSCA --out_dropout=0.1 --patch_size_stride="2 2" --patience=20 --root_path=./datasets/m4 --seasonal_patterns=Weekly --seed=42 --split_len=2 --task_name=short_term_forecast --test_flag=0 --train_epochs=100 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1 --wd_project=fsca-st-log00

# daily
CUDA_VISIBLE_DEVICES=0 python run.py --batch_size=16 --c_out=1 --d_ff=512 --d_model=768 --data=m4 --dec_in=1 --des=Exp --enc_in=1 --features=M --in_dropout=0 --is_training=1 --itr=1 --layer_index=4*0*0 --learning_rate=0.001 --loss=SMAPE --model=FSCA --out_dropout=0.1 --patch_size_stride="4 4" --patience=20 --root_path=./datasets/m4 --seasonal_patterns=Daily --seed=42 --split_len=2 --task_name=short_term_forecast --test_flag=0 --train_epochs=100 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1 --wd_project=fsca-st-log00

# hour
CUDA_VISIBLE_DEVICES=0 python run.py --batch_size=16 --c_out=1 --d_ff=128 --d_model=768 --data=m4 --dec_in=1 --des=Exp --enc_in=1 --features=M --in_dropout=0 --is_training=1 --itr=1 --layer_index=4*0*0 --learning_rate=0.001 --loss=SMAPE --model=FSCA --out_dropout=0.1 --patch_size_stride="2 2" --patience=20 --root_path=./datasets/m4 --seasonal_patterns=Hourly --seed=42 --split_len=2 --task_name=short_term_forecast --test_flag=0 --train_epochs=100 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1 --wd_project=fsca-st-log00