datename=$(date +%Y%m%d-%H%M%S)
data_name=finland
country=Finland

python test_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --country $country \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/checkpoints/finland_with_sam_ours_epoch41_bestRsum0.6839.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
