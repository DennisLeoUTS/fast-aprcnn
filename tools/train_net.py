#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
import os

#os.environ['GLOG_minloglevel'] = '3'

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver', help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='cub_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--partid', dest='partid',
                        help='part id for part-based methods', default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class fake:
    def __init__(self, name):
        self.name = name


if __name__ == '__main__':
    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
        
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)
        
        
    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    print 'End loading roidb'

    if imdb.name.startswith('cub') or imdb.name.startswith('nabirds'):
        i = args.partid
        fake_net = fake(imdb.parts[i])
        output_dir = get_output_dir(imdb, fake_net)
        print 'Output will be saved to `{:s}`'.format(output_dir)
        
        train_net(args.solver, roidb[i], output_dir,
                pretrained_model=args.pretrained_model,
                max_iters=args.max_iters)
    else:
        output_dir = get_output_dir(imdb, None)
        print 'Output will be saved to `{:s}`'.format(output_dir)
        train_net(args.solver, roidb, output_dir,
                  pretrained_model=args.pretrained_model,
                  max_iters=args.max_iters)
