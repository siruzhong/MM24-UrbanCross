datename=$(date +%Y%m%d-%H%M%S)
# country_source=Finland
# country=Spain
country=Finland
# lr=5e-7
name=urbancross_zeroshot_$country
logging_dir=outputs/zeroshot/$country/$name/$datename
mkdir -p $logging_dir
pip list > $logging_dir/environment.txt
cp $0 $logging_dir/$(basename $0 .sh).sh
python zeroshot_urbancross.py \
       --gpuid 0 \
       --model_name $name \
       --experiment_name $name \
       --ckpt_save_path $logging_dir \
       --wandb_logging_dir $logging_dir \
       --image_path urbancross_data/images_target \
       --batch_size_test 80 \
       --country $country \
       --load_path outputs/pretrain/Finland/urbancross_Finland_bs40_lr1e-5_numseg5/20240404-004033/ckpt_urbancross_Finland_bs40_lr1e-5_numseg5_19_0.36.pth \
       --workers 0 \
       |& tee $logging_dir/logs_$datename.txt 2>&1
       # --load_path outputs/finetune/Finland_2_Spain/20240331-200543/ckpt_urbancross_finetune_Finland_2_Spain_lr5e-7_3_0.04.pth \
       # --lr $lr \
       # --load_path outputs/pretrain/Finland/20240329-160815/ckpt_urbancross_Finland_bs40_lr1e-5_numseg5_41_0.37.pth \
       # --epochs 50 \
       # --k_fold_nums 1 \
       # --country_target $country_target \
       # --batch_size_target 5 \
       # --batch_size_val 10\
       # --data_name rsitmd  \