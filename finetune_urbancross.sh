datename=$(date +%Y%m%d-%H%M%S)
# RSITMD Dataset
python finetune_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --ckpt_save_path outputs/checkpoints/ \
       --epochs 50 \
       --k_fold_nums 1 \
       --image_path urbancross_data/images_target \
       --batch_size_source 10 \
       --batch_size_target 5 \
       --country_source Finland \
       --country_target Spain \
       --workers 0 \
       |& tee outputs/logs_$datename.txt 2>&1
       # --batch_size_target 5 \
       # --batch_size_val 10\
       # --data_name rsitmd  \