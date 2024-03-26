datename=$(date +%Y%m%d-%H%M%S)
# RSICD Dataset
python train_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --ckpt_save_path outputs/checkpoints/ \
       --epochs 35 \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/images \
       --country "" \
       --batch_size 40 \
       --num_seg 4 \
       --workers 0 \
       --data_name rsicd \
       --test_step 35 \
       2>&1 | tee -a outputs/logs_$datename.txt