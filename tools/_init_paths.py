# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

main_dir = osp.join(this_dir, '..')

# Add linear to PYTHONPATH
# liblinear_path = osp.join(this_dir, 'utils', 'liblinear', 'python')
# add_path(liblinear_path)

# Add caffe to PYTHONPATH
caffe_path = osp.join(main_dir, '..', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(main_dir, 'lib')
add_path(lib_path)
