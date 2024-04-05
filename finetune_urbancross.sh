datename=$(date +%Y%m%d-%H%M%S)
country_source=Finland
country_target=Spain
lr=1e-6
name=urbancross_finetune_$country_source\_2_$country_target\_lr$lr
logging_dir=outputs/finetune/$country_source\_2_$country_target/$name/$datename
mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh
python finetune_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --epochs 10 \
       --lr $lr \
       --image_path urbancross_data/images_target \
       --batch_size_source 80 \
       --batch_size_target 40 \
       --country_source $country_source \
       --country_target $country_target \
       --load_path outputs/pretrain/Finland/urbancross_Finland_bs40_lr1e-5_numseg5/20240404-004033/ckpt_urbancross_Finland_bs40_lr1e-5_numseg5_19_0.36.pth \
       |& tee $logging_dir/logs_$datename.txt 2>&1
       # --k_fold_nums 1 \
       # --workers 0 \
       # --batch_size_target 5 \
       # --batch_size_val 10\
       # --data_name rsitmd  \