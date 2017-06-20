#!/bin/bash
# part id: $1,  gpu id: $2
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/cub_svm_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_svms_cub_detector.py --gpu $2 \
  --def models/VGG16_cub/test.prototxt \
  --net output/default/cub_train/{:s}/vgg16_fast_rcnn_iter_40000.caffemodel \
  --imdb cub_train \
  --cfg experiments/cfgs/svm.yml \
  --partid $1

