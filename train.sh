TIME=$(date "+%Y%m%d-%H%M%S")
INFO="att"
FILENAME=train_logs/train_${INFO}_${TIME}_log
echo ${FILENAME}
CUDA_VISIBLE_DEVICES=3 nohup python main.py > ${FILENAME} 2>&1 &