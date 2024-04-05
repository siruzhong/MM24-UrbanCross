datename=$(date +%Y%m%d-%H%M%S)
name=urbancross_finetune
country_source=Finland
country_target=Spain
logging_dir=outputs/finetune/$country_source\_2_$country_target/$datename
mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh
python finetune_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --epochs 50 \
       --k_fold_nums 1 \
       --lr 2e-4 \
       --image_path urbancross_data/images_target \
       --batch_size_source 80 \
       --batch_size_target 40 \
       --country_source $country_source \
       --country_target $country_target \
       --load_path outputs/checkpoints/urbancrossckpt_urbancross_38_0.47.pth \
       |& tee $logging_dir/logs_$datename.txt 2>&1
       # --workers 0 \
       # --batch_size_target 5 \
       # --batch_size_val 10\
       # --data_name rsitmd  \