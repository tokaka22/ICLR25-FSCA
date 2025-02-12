. activate FSCA

path=/code/TimeLLM/code/ICLR25-FSCA-official/Zero-shot_Learning
wd_project=fsca-zt-log00
gpu_id=0

echo $path
echo $wd_project
cd $path

# 96
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --wd_project=$wd_project --batch_size=256 --c_out=7 --d_ff=512 --d_l_comp=512 --d_model=768 --data_flow=ETTh1_ETTm2 --decay_fac=0.5 --edge_eta_min=1e-08 --enc_in=7 --eta_min=1e-08 --features=M --freq=0 --in_dropout=0 --is_gpt=1 --itr=1 --layer_index=4*0_3_4*0_3_4 --learning_rate=0.0005 --lradj=COS --model=FSCA --multi=1 --out_dropout=0.1 --patch_size_stride="48 48" --patience=5 --percent=100 --pred_len=96 --revin_flag=0 --seed=42 --seq_len=512 --split_len=2 --sweep_flag=1 --test_flag=0 --tmax=20 --train_epochs=100 --train_shuffle_int=1 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1

# 192
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --wd_project=$wd_project --batch_size=256 --c_out=7 --d_ff=256 --d_l_comp=256 --d_model=768 --data_flow=ETTh1_ETTm2 --decay_fac=0.5 --edge_eta_min=1e-08 --enc_in=7 --eta_min=1e-08 --features=M --freq=0 --in_dropout=0 --is_gpt=1 --itr=1 --layer_index=4*0_3_4*0_3_4 --learning_rate=0.0005 --lradj=COS --model=FSCA --multi=1 --out_dropout=0.1 --patch_size_stride="48 48" --patience=5 --percent=100 --pred_len=192 --revin_flag=0 --seed=42 --seq_len=512 --split_len=2 --sweep_flag=1 --test_flag=0 --tmax=20 --train_epochs=100 --train_shuffle_int=1 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1

# 336
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --wd_project=$wd_project --batch_size=256 --c_out=7 --d_ff=512 --d_l_comp=512 --d_model=768 --data_flow=ETTh1_ETTm2 --decay_fac=0.5 --edge_eta_min=1e-08 --enc_in=7 --eta_min=1e-08 --features=M --freq=0 --in_dropout=0 --is_gpt=1 --itr=1 --layer_index=4*0_3_4*0_3_4 --learning_rate=0.0005 --lradj=COS --model=FSCA --multi=1 --out_dropout=0.1 --patch_size_stride="48 48" --patience=5 --percent=100 --pred_len=336 --revin_flag=0 --seed=42 --seq_len=512 --split_len=2 --sweep_flag=1 --test_flag=0 --tmax=20 --train_epochs=100 --train_shuffle_int=1 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1

# 720
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --wd_project=$wd_project --batch_size=256 --c_out=7 --d_ff=512 --d_l_comp=512 --d_model=768 --data_flow=ETTh1_ETTm2 --decay_fac=0.5 --edge_eta_min=1e-08 --enc_in=7 --eta_min=1e-08 --features=M --freq=0 --in_dropout=0 --is_gpt=1 --itr=1 --layer_index=4*0_3_4*0_3_4 --learning_rate=0.0005 --lradj=COS --model=FSCA --multi=1 --out_dropout=0.1 --patch_size_stride="48 48" --patience=5 --percent=100 --pred_len=720 --revin_flag=0 --seed=42 --seq_len=512 --split_len=2 --sweep_flag=1 --test_flag=0 --tmax=20 --train_epochs=100 --train_shuffle_int=1 --w_l2s_flag=0 --w_l2s_v=0.0001 --wandb_flag=1