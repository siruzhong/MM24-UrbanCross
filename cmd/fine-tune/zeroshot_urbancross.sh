datename=$(date +%Y%m%d-%H%M%S)
country=Spain
name=urbancross_zeroshot_$country
logging_dir=outputs/zeroshot/$country/$name/$datename

cd ../../

mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh

python zeroshot_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target \
       --batch_size_test 80 \
       --country $country \
       --load_path /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_finland/checkpoints/finland_with_sam_ours_epoch15_bestRsum0.7644.pth \
       --workers 0 \
       2>&1 | tee -a $logging_dir/logs_$datename.txt