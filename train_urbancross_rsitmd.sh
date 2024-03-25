datename=$(date +%Y%m%d-%H%M%S)
# RSITMD Dataset
python train_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --ckpt_save_path outputs/checkpoints/ \
       --epochs 40 \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/images \
       --country "" \
       --batch_size 40 \
       --num_seg 1 \
       --workers 0 \
       --data_name rsitmd \
       --test_step 40 \
       2>&1 | tee -a outputs/logs_$datename.txt