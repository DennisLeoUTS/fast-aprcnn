#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
import cv2
import numpy as np
from fast_rcnn.test import test_net_cub, nms
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import matplotlib.pyplot as plt
import cPickle


def vis_detections(im, imname, selected, dets, thresh=0.3):
    """Visual debugging of detections."""
    im = im[:, :, (2, 1, 0)]
    # np.array(.astype(np.float32, copy=False)
    plt.cla()
    plt.imshow(im)
    plt.gca().hold()
    colors = ['g', 'r', 'b']
    scores = []
    for k in xrange(len(dets)):
        bbox = dets[k][:4]
        score = dets[k][-1]
        scores.append(score)
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[k], linewidth=3)
            )
        
    plt.title('{}  {:.3f}-{}, {:.3f}-{}, {:.3f}-{}'.format(imname,\
            scores[0], selected[0], scores[1], selected[1], scores[2], selected[2]))
    plt.show()
    
    
def save_detections(im, imind, outdir, dets, thresh=0.3):
    """Visual debugging of detections."""
    im = im[:, :, (2, 1, 0)]
    # np.array(.astype(np.float32, copy=False)
    plt.cla()
    plt.imshow(im)
    plt.gca().hold()
    colors = ['g', 'r', 'b']
    scores = []
    for k in xrange(len(dets)):
        bbox = dets[k][:4]
        score = dets[k][-1]
        scores.append(score)
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[k], linewidth=3)
            )
        
    plt.title('{}  {:.3f}, {:.3f}, {:.3f}'.format(imind, scores[0], scores[1], scores[2]))
    #plt.show()
    plt.savefig(os.path.join(outdir, '{:05d}.png'.format(imind)))  


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='View detection results')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='experiments/cfgs/nabirds_svm.yml', type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='nabirds_train', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--partid', dest='partid',
                        help='part id for part-based methods', default=0, type=int)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


class fake:
    def __init__(self, name):
        self.name = name

if __name__ == '__main__':
    args = parse_args()
    
    #args.imdb_name = 'nabirds_test'
    args.imdb_name = 'cub_weak'
    args.cfg_file = 'experiments/cfgs/svm.yml'

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    

    imdb = get_imdb(args.imdb_name)
    num_images = len(imdb.image_index)
    names = imdb.parts
    print names
    
    
    boxes_all = []
    f = fake('vgg16_fast_rcnn_iter_40000_svm')
    output_dir = get_output_dir(imdb, f)
    for partid in xrange(imdb.num_parts):
        det_file = os.path.join(output_dir, imdb.parts[partid] + '_detections.pkl')

        with open(det_file, 'rb') as f:
            boxes = cPickle.load(f)
            boxes_all.append(boxes)
    
    cache_file = os.path.join(cfg.PRCNN.WEAK_DIR, 'selected_samples_v2.pkl')
    with open(cache_file, 'rb') as fid:
        selected_inds = cPickle.load(fid)
        print len(selected_inds), len(selected_inds[0])
    
    P = np.random.permutation(num_images)
    #outdir = os.path.join('/DB/rhome/zxu/workspace/fast-rcnn/fast-rcnn/data/figure/', imdb.name)
    
    for j in xrange(num_images):
        i = P[j]
        #i = j
        im = cv2.imread(imdb.image_path_at(i))
        imboxes = []
        flag = True
        for k in xrange(imdb.num_parts):
            if len(boxes_all[k][1][i])==0:
                flag = False
                break
            box = boxes_all[k][1][i][0, :]
            imboxes.append(box)
            
        if flag:
            #vis_detections(im, i, imboxes, -1000)
            print 'Image {}'.format(i), imboxes
            vis_detections(im, i, [selected_inds[s].count(i) for s in range(imdb.num_parts)], imboxes, -1000)
