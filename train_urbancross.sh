datename=$(date +%Y%m%d-%H%M%S)
# RSITMD Dataset
python train_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name ours \
       --data_name rsitmd  \
       --ckpt_save_path checkpoint/ \
       --epochs 50 \
       --batch_size 5 \
       --k_fold_nums 1 \
       --workers 0 \
       --image_path urbancross_data/images_target \
       |& tee outputs/logs_$datename.txt 2>&1