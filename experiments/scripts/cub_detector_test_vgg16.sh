#!/bin/bash
#$1: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/cub_detector_test_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16_cub/test.prototxt \
  --net output/default/cub_train/{:s}/vgg16_fast_rcnn_iter_40000_svm.caffemodel \
  --cfg experiments/cfgs/svm.yml \
  --imdb cub_test


time ./tools/test_net.py --gpu $1 \
  --def models/VGG16_cub/test.prototxt \
  --net output/default/cub_train/{:s}/vgg16_fast_rcnn_iter_40000_svm.caffemodel \
  --cfg experiments/cfgs/svm.yml \
  --imdb cub_weak

