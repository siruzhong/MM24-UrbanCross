datename=$(date +%Y%m%d-%H%M%S)
data_name=finland
country=Finland

cd ../../

python test_urbancross_without_sam.py \
       --gpuid 0 \
       --model_name urbancross \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --country $country \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_germany/checkpoints/germany_without_sam_ours_epoch22_bestRsum0.8133.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
