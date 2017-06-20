#!/bin/bash
#$1: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/cub_retrain_prcnn_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_prcnn.py --gpu $1 \
  --net output/retrain/cub_weak/{:s}/vgg16_fast_rcnn_iter_40000_svm.caffemodel \
  --def models/VGG16_cub/test.prototxt \
  --imdb_train cub_weak \
  --imdb_test cub_test \
  --cfg experiments/cfgs/retrain_prcnn.yml \
  
  




