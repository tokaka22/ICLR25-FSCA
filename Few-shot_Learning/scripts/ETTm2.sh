path=/code/TimeLLM/code/ICLR25-FSCA-official/Few-shot_Learning
wd_project=fsca-fs-log00
gpu_id=0

cd $path/scripts/ETTm2
bash ./96.sh $path $wd_project $gpu_id
bash ./192.sh $path $wd_project $gpu_id
bash ./336.sh $path $wd_project $gpu_id
bash ./720.sh $path $wd_project $gpu_id