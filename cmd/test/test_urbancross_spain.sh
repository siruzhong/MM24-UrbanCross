datename=$(date +%Y%m%d-%H%M%S)
data_name=spain
country=Spain

cd ../../

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
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_finland/checkpoints/finland_with_sam_ours_epoch15_bestRsum0.7644.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
