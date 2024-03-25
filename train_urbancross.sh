datename=$(date +%Y%m%d-%H%M%S)
name=urbancross
# country=Finland
country=Germany
logging_dir=outputs/pretrain/$country/$datename
mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh
python train_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --epochs 50 \
       --image_path urbancross_data/images_target \
       --country $country \
       --batch_size 40 \
       --num_seg 5 \
       |& tee $logging_dir/logs_$datename.txt 2>&1
       # --workers 0 \
       # --k_fold_nums 1 \
       # --batch_size_val 10\
       # --data_name rsitmd  \

# datename=$(date +%Y%m%d-%H%M%S)
# name=urbancross
# python train_urbancross.py \
#        --gpuid 0 \
#        --model_name $name \
#        --experiment_name $name \
#        --ckpt_save_path outputs/checkpoints/pretrain \
#        --epochs 50 \
#        --image_path urbancross_data/images_target \
#        --country Finland \
#        --batch_size 40 \
#        --num_seg 5 \
#        |& tee outputs/pretrain/logs_$datename.txt 2>&1
#        # --resume outputs/checkpoints/urbancrossckpt_urbancross_38_0.47.pth \
#        # --workers 0 \
#        # --k_fold_nums 1 \
#        # --batch_size_val 10\
#        # --data_name rsitmd  \