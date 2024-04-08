datename=$(date +%Y%m%d-%H%M%S)
data_name=rsitmd

cd ../../

python test_urbancross_without_sam.py \
       --gpuid 0 \
       --model_name urbancross \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/images \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/new_00_rsitmd/checkpoints/rsitmd_without_sam_ours_epoch5_bestRsum0.6200.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
