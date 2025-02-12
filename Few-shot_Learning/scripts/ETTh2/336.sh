source activate FSCA

path=$1
wd_project=$2
gpu_id=$3

echo $path
echo $wd_project
cd $path

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --wd_project=$wd_project \
    --batch_size=256 \
    --c_out=7 \
    --d_ff=32 \
    --d_l_comp=512 \
    --d_model=768 \
    --data=ETTh2 \
    --data_path=nan \
    --decay_fac=0.5 \
    --edge_eta_min=1e-08 \
    --enc_in=7 \
    --eta_min=1e-08 \
    --features=M \
    --freq=0 \
    --in_dropout=0 \
    --is_gpt=1 \
    --itr=1 \
    --layer_index=4*0*0 \
    --learning_rate=0.0001 \
    --lradj=COS \
    --model=FSCA \
    --multi=0 \
    --out_dropout=0.1 \
    --patch_size_stride="8 8" \
    --patience=10 \
    --percent=5 \
    --pred_len=336 \
    --revin_flag=0 \
    --root_path=nan \
    --seed=42 \
    --seq_len=336 \
    --split_len=2 \
    --sweep_flag=1 \
    --test_flag=0 \
    --tmax=20 \
    --train_epochs=50 \
    --train_shuffle_int=1 \
    --w_l2s_flag=0 \
    --w_l2s_v=0.0001 \
    --wandb_flag=1