datename=$(date +%Y%m%d-%H%M%S)
data_name=rsitmd

python test_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/images \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_rsitmd/checkpoints/seg6/rsitmd_with_sam_ours_epoch6_bestRsum0.6161.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
