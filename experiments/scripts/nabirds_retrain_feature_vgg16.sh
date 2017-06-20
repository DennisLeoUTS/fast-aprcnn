#!/bin/bash
#$1: part id, $2: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/nabirds_retrain_feature_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $2 \
  --solver models/VGG16_nabirds/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb nabirds_train \
  --iters 100000 \
  --cfg experiments/cfgs/nabirds_retrain_feature.yml \
  --partid $1