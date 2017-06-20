#!/bin/bash
#$1: part id, $2: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/cub_retrain_detector_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/test_net.py --gpu $2 \
  --def models/VGG16_cub/test.prototxt \
  --net output/retrain/cub_weak/{:s}/vgg16_fast_rcnn_iter_40000_svm.caffemodel \
  --cfg experiments/cfgs/retrain_svm.yml \
  --imdb cub_test\
  --partid $1

time ./tools/test_net.py --gpu $2 \
  --def models/VGG16_cub/test.prototxt \
  --net output/retrain/cub_weak/{:s}/vgg16_fast_rcnn_iter_40000_svm.caffemodel \
  --cfg experiments/cfgs/retrain_svm.yml \
  --imdb cub_weak\
  --partid $1

