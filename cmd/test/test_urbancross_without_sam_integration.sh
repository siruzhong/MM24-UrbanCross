datename=$(date +%Y%m%d-%H%M%S)
data_name=integration
country=Integration

cd ../../

python test_urbancross_without_sam.py \
       --gpuid 0 \
       --model_name urbancross \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --country $country \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_integration/checkpoints/integration_without_sam_ours_epoch45_bestRsum0.6633.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
