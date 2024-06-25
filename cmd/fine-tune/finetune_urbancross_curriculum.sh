datename=$(date +%Y%m%d-%H%M%S)
country_source=Finland
country_target=Spain
lr=1e-7
name=urbancross_finetune_curriculum_$country_source\_2_$country_target\_lr$lr
logging_dir=outputs/finetune_curriculum/$country_source\_2_$country_target/$datename

cd ../../

mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh

python finetune_urbancross_curriculum.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --epochs 5 \
       --lr $lr \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross \
       --batch_size_source 80 \
       --batch_size_target 16 \
       --country_source $country_source \
       --country_target $country_target \
       --load_path /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_finland/checkpoints/finland_with_sam_ours_epoch11_bestRsum0.7556.pth \
       2>&1 | tee -a $logging_dir/logs_$datename.txt