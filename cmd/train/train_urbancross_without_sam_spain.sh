datename=$(date +%Y%m%d-%H%M%S)
data_name=spain
country=Spain
epochs=15

cd ../../

python train_urbancross_without_sam.py \
       --gpuid 0 \
       --model_name urbancross \
       --experiment_name urbancross \
       --ckpt_save_path outputs/checkpoints/ \
       --epochs $epochs \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross \
       --country $country \
       --batch_size 45 \
       --lr 0.00001 \
       --workers 0 \
       --data_name $data_name \
       --test_step $epochs \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_epochs${epochs}_without_sam.txt
