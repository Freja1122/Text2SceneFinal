TIME=$(date "+%Y%m%d-%H%M%S")
INFO="att"
FILENAME=train_logs/test_${INFO}_${TIME}_log
echo ${FILENAME}
nohup python main.py > ${FILENAME} 2>&1 &