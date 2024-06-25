datename=$(date +%Y%m%d-%H%M%S)
data_name=spain
country=Spain

cd ../../

python test_urbancross.py \
       --gpuid 0 \
       --model_name urbancross \
       --experiment_name urbancross \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/UrbanCross \
       --country "" \
       --batch_size 40 \
       --workers 0 \
       --country $country \
       --data_name $data_name \
       --resume /hpc2hdd/home/szhong691/zsr/projects/UrbanCross/outputs/finetune/Finland_2_Spain/02_ratio_0.2_lr_1e-7_finland_epoch_11/20240406-193304ckpt_urbancross_finetune_Finland_2_Spain_lr1e-7_3_0.0462.pth \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_test.txt
