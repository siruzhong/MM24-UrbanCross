datename=$(date +%Y%m%d-%H%M%S)
country=Finland
# country=Germany
bs=40
# lr=1e-4
# lr=5e-5
lr=1e-5
# lr=5e-6
num_seg=5
name=urbancross_$country\_bs$bs\_lr$lr\_numseg$num_seg
logging_dir=outputs/pretrain/$country/$name/$datename
mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh
python train_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --epochs 30 \
       --image_path urbancross_data/images_target \
       --country $country \
       --batch_size $bs \
       --lr $lr \
       --num_seg $num_seg \
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