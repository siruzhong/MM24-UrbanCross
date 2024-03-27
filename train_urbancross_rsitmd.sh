datename=$(date +%Y%m%d-%H%M%S)
data_name=rsitmd
epochs=30
num_seg=5

python train_urbancross.py \
       --gpuid 0 \
       --model_name ours \
       --experiment_name urbancross \
       --ckpt_save_path outputs/checkpoints/ \
       --epochs $epochs \
       --image_path /hpc2hdd/home/szhong691/zsr/projects/dataset/RSITMD/images \
       --country "" \
       --batch_size 40 \
       --num_seg $num_seg \
       --workers 0 \
       --lr 0.0001 \
       --data_name $data_name \
       --test_step $epochs \
       2>&1 | tee -a outputs/logs_${datename}_${data_name}_epochs${epochs}_seg${num_seg}_with_sam.txt
