#!/bin/bash
#$1: part id, $2: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/cub_default_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $2 \
  --solver models/VGG16_cub/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb cub_train \
  --partid $1


#time ./tools/test_net.py --gpu $2 \
#  --def models/VGG16_cub/test.prototxt \
#  --net output/default/cub_train/{:s}/vgg16_fast_rcnn_iter_40000.caffemodel \
#  --imdb cub_test
#  --partid $1
