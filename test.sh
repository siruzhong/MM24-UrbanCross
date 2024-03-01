export TF_ENABLE_ONEDNN_OPTS=0
datename=$(date +%Y%m%d-%H%M%S)
python test_single.py \
       --resume checkpoint/rsitmd/SWAN/SWAN_best.pth.tar \
       |& tee outputs/logs_test_$datename.txt 2>&1