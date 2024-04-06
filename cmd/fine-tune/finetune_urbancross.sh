datename=$(date +%Y%m%d-%H%M%S)
name=urbancross_finetune
country_source=Finland
country_target=Spain
logging_dir=outputs/finetune/$country_source\_2_$country_target/$datename

cd ../../

mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh

python finetune_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --epochs 5 \
       --k_fold_nums 1 \
       --lr 0.00001 \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target \
       --batch_size_source 80 \
       --batch_size_target 40 \
       --country_source $country_source \
       --country_target $country_target \
       --load_path /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_finland/checkpoints/finland_with_sam_ours_epoch15_bestRsum0.7644.pth \
       2>&1 | tee -a $logging_dir/logs_$datename.txt
       # --workers 0 \
       # --batch_size_target 5 \
       # --batch_size_val 10\
       # --data_name rsitmd  \