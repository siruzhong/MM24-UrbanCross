datename=$(date +%Y%m%d-%H%M%S)
# RSICD Dataset
python test_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/RSICD/images \
       --country "" \
       --batch_size 40 \
       --num_seg 5 \
       --workers 0 \
       --data_name rsicd \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/checkpoints/ckpt_rsicd_0.24555555793146291.pth \
       2>&1 | tee -a outputs/logs_$datename.txt