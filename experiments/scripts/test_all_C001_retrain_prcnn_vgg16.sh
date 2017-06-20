#!/bin/bash
#$1: gpu id
set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/est_part_C001_retrain_prcnn_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_prcnn.py --gpu $1 \
  --net output/retrain/cub_weak/{:s}/vgg16_fast_rcnn_iter_40000.caffemodel \
  --detpath output/svm/{:s}/vgg16_fast_rcnn_iter_40000_svm/{:s}_detections.pkl \
  --def models/VGG16_cub/test.prototxt \
  --imdb_train cub_train \
  --imdb_test cub_test \
  --imdb_weak cub_weak \
  --cfg experiments/cfgs/baseline_oridetect_retrain_prcnn.yml \
  --mode all \
  --C 0.001
  
  




