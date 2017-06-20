#!/bin/bash
#$1: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/nabirds_retrain_prcnn_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_prcnn.py --gpu $1 \
  --net output/nabirds/nabirds_train/{:s}/vgg16_fast_rcnn_iter_100000.caffemodel \
  --detpath output/nabirds/{:s}/vgg16_fast_rcnn_iter_40000_svm/{:s}_detections.pkl \
  --def models/VGG16_nabirds/test.prototxt \
  --imdb_train nabirds_train \
  --imdb_test nabirds_test \
  --imdb_weak nabirds_train \
  --cfg experiments/cfgs/nabirds_retrain_prcnn.yml
  
  




