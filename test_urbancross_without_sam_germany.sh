datename=$(date +%Y%m%d-%H%M%S)
data_name=germany
country=Germany

python test_urbancross_without_sam.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross/image_target \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --country $country \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/checkpoints/germany_without_sam_ours_epoch44_bestRsum0.7239.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
