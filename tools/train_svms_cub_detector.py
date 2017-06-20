#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Train post-hoc SVMs using the algorithm and hyper-parameters from
traditional R-CNN.
Train part detectors for cub
"""

import _init_paths
import os
os.environ['GLOG_minloglevel'] = '3'

from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from fast_rcnn.test import im_detect
from utils.timer import Timer
import caffe
import argparse
import pprint
import numpy as np
import numpy.random as npr
import cv2
from sklearn import svm
import sys
import os.path as osp
import cPickle

class SVMTrainer(object):
    """
    Trains post-hoc detection SVMs for all classes using the algorithm
    and hyper-parameters of traditional R-CNN.
    """

    def __init__(self, net, imdb, partid):
        self.imdb = imdb
        self.net = net
        self.layer = 'fc7'
        self.part = imdb.parts[partid]
        self.partid = partid
        self.hard_thresh = -1.0001
        self.neg_iou_thresh = 0.3

        dim = net.params['cls_score'][0].data.shape[1]
        scale = self._get_feature_scale()
        print('Feature dim: {}'.format(dim))
        print('Feature scale: {:.3f}'.format(scale))
        parts = imdb.parts[partid:partid+1]
        parts.insert(0, '__background__')
        
        if cfg.PRCNN.NOISE_REMOVE:
            print 'Noise removal conducted, load selected samples after noise removal'
            cache_file = os.path.join(cfg.PRCNN.WEAK_DIR, 'selected_samples.pkl')
            with open(cache_file, 'rb') as f:
                selected_inds_all = cPickle.load(f)
                self.selected_inds = selected_inds_all[partid]
                print 'Selected_inds', len(self.selected_inds), sum(self.selected_inds)
        
        self.trainers = [SVMClassTrainer(cls, dim, feature_scale=scale)
                         for cls in parts]

    def _get_feature_scale(self, num_images=100):
        TARGET_NORM = 20.0 # Magic value from traditional R-CNN
        _t = Timer()
        roidb = self.imdb.roidb
        total_norm = 0.0
        count = 0.0
        inds = npr.choice(xrange(self.imdb.num_images), size=num_images,
                          replace=False)
        for i_, i in enumerate(inds):
            im = cv2.imread(self.imdb.image_path_at(i))
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            _t.tic()
            scores, boxes = im_detect(self.net, im, roidb[i]['boxes'])
            _t.toc()
            feat = self.net.blobs[self.layer].data
            total_norm += np.sqrt((feat ** 2).sum(axis=1)).sum()
            count += feat.shape[0]
            print('{}/{}: avg feature norm: {:.3f}'.format(i_ + 1, num_images,
                                                           total_norm / count))

        return TARGET_NORM * 1.0 / (total_norm / count)

    def _get_pos_counts(self):
        counts = np.zeros((2), dtype=np.int)
        roidb = self.imdb.roidb
        for i in xrange(len(roidb)):
            if cfg.PRCNN.NOISE_REMOVE and self.selected_inds.count(i)==0:
                continue
            I = np.where(roidb[i]['gt_classes'] == self.partid+1)[0]   # 0 for background
            counts[1] += len(I)

        print('class {:s} has {:d} positives'.
                format(self.part, counts[1]))

        return counts

    def get_pos_examples(self):
        counts = self._get_pos_counts()
        for i in xrange(len(counts)):
            self.trainers[i].alloc_pos(counts[i])

        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)
        #num_images = 100
        for i in xrange(num_images):
            if cfg.PRCNN.NOISE_REMOVE and self.selected_inds.count(i)==0:
                continue
            im = cv2.imread(self.imdb.image_path_at(i))
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            gt_inds = np.where(roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = roidb[i]['boxes'][gt_inds]
            _t.tic()
            scores, boxes = im_detect(self.net, im, gt_boxes)
            _t.toc()
            feat = self.net.blobs[self.layer].data

            cls_inds = np.where(roidb[i]['gt_classes'][gt_inds] == self.partid+1)[0]
            if len(cls_inds) > 0:
                cls_feat = feat[cls_inds, :]
                self.trainers[1].append_pos(cls_feat)

            print 'get_pos_examples: {:d}/{:d} {:.3f}s' \
                  .format(i + 1, len(roidb), _t.average_time)

    def initialize_net(self):
        # Start all SVM parameters at zero
        self.net.params['cls_score'][0].data[...] = 0
        self.net.params['cls_score'][1].data[...] = 0

        # Initialize SVMs in a smart way. Not doing this because its such
        # a good initialization that we might not learn something close to
        # the SVM solution.
#        # subtract background weights and biases for the foreground classes
#        w_bg = self.net.params['cls_score'][0].data[0, :]
#        b_bg = self.net.params['cls_score'][1].data[0]
#        self.net.params['cls_score'][0].data[1:, :] -= w_bg
#        self.net.params['cls_score'][1].data[1:] -= b_bg
#        # set the background weights and biases to 0 (where they shall remain)
#        self.net.params['cls_score'][0].data[0, :] = 0
#        self.net.params['cls_score'][1].data[0] = 0

    def update_net(self, cls_ind, w, b):
        self.net.params['cls_score'][0].data[cls_ind, :] = w
        self.net.params['cls_score'][1].data[cls_ind] = b

    def train_with_hard_negatives(self):
        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)
        #num_images = 100
        for i in xrange(num_images):
            if cfg.PRCNN.NOISE_REMOVE and self.selected_inds.count(i)==0:
                continue
            im = cv2.imread(self.imdb.image_path_at(i))
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            _t.tic()
            scores, boxes = im_detect(self.net, im, roidb[i]['boxes'])
            _t.toc()
            feat = self.net.blobs[self.layer].data

            hard_inds = \
                np.where((scores[:, self.partid+1] > self.hard_thresh) &
                        (roidb[i]['gt_overlaps'][:, self.partid].ravel() <
                        self.neg_iou_thresh))[0]
            if len(hard_inds) > 0:
                hard_feat = feat[hard_inds, :].copy()
                new_w_b = \
                    self.trainers[1].append_neg_and_retrain(feat=hard_feat)
                if new_w_b is not None:
                    self.update_net(1, new_w_b[0], new_w_b[1])

            print(('train_with_hard_negatives: '
                   '{:d}/{:d} {:.3f}s').format(i + 1, len(roidb),
                                               _t.average_time))

    def train(self):
        # Initialize SVMs using
        #   a. w_i = fc8_w_i - fc8_w_0
        #   b. b_i = fc8_b_i - fc8_b_0
        #   c. Install SVMs into net
        self.initialize_net()

        # Pass over roidb to count num positives for each class
        #   a. Pre-allocate arrays for positive feature vectors
        # Pass over roidb, computing features for positives only
        self.get_pos_examples()

        # Pass over roidb
        #   a. Compute cls_score with forward pass
        #   b. For each class
        #       i. Select hard negatives
        #       ii. Add them to cache
        #   c. For each class
        #       i. If SVM retrain criteria met, update SVM
        #       ii. Install new SVM into net
        self.train_with_hard_negatives()

        # One final SVM retraining for each class
        # Install SVMs into net
        new_w_b = self.trainers[1].append_neg_and_retrain(force=True)
        self.update_net(1, new_w_b[0], new_w_b[1])

class SVMClassTrainer(object):
    """Manages post-hoc SVM training for a single object class."""

    def __init__(self, cls, dim, feature_scale=1.0,
                 C=0.001, B=10.0, pos_weight=2.0):
        self.pos = np.zeros((0, dim), dtype=np.float32)
        self.neg = np.zeros((0, dim), dtype=np.float32)
        self.B = B
        self.C = C
        self.cls = cls
        self.pos_weight = pos_weight
        self.dim = dim
        self.feature_scale = feature_scale
        self.svm = svm.LinearSVC(C=C, class_weight={1: 2, -1: 1},
                                 intercept_scaling=B, verbose=1,
                                 penalty='l2', loss='l1',
                                 random_state=cfg.RNG_SEED, dual=True)
        self.pos_cur = 0
        self.num_neg_added = 0
        self.retrain_limit = 2000
        self.evict_thresh = -1.1
        self.loss_history = []

    def alloc_pos(self, count):
        self.pos_cur = 0
        self.pos = np.zeros((count, self.dim), dtype=np.float32)

    def append_pos(self, feat):
        num = feat.shape[0]
        self.pos[self.pos_cur:self.pos_cur + num, :] = feat
        self.pos_cur += num

    def train(self):
        print('>>> Updating {} detector <<<'.format(self.cls))
        num_pos = self.pos.shape[0]
        num_neg = self.neg.shape[0]
        print('Cache holds {} pos examples and {} neg examples'.
              format(num_pos, num_neg))
        X = np.vstack((self.pos, self.neg)) * self.feature_scale
        y = np.hstack((np.ones(num_pos),
                       -np.ones(num_neg)))
        self.svm.fit(X, y)
        w = self.svm.coef_
        b = self.svm.intercept_[0]
        scores = self.svm.decision_function(X)
        pos_scores = scores[:num_pos]
        neg_scores = scores[num_pos:]

        pos_loss = (self.C * self.pos_weight *
                    np.maximum(0, 1 - pos_scores).sum())
        neg_loss = self.C * np.maximum(0, 1 + neg_scores).sum()
        reg_loss = 0.5 * np.dot(w.ravel(), w.ravel()) + 0.5 * b ** 2
        tot_loss = pos_loss + neg_loss + reg_loss
        self.loss_history.append((tot_loss, pos_loss, neg_loss, reg_loss))

        for i, losses in enumerate(self.loss_history):
            print(('    {:d}: obj val: {:.3f} = {:.3f} '
                   '(pos) + {:.3f} (neg) + {:.3f} (reg)').format(i, *losses))

        return ((w * self.feature_scale, b * self.feature_scale),
                pos_scores, neg_scores)

    def append_neg_and_retrain(self, feat=None, force=False):
        if feat is not None:
            num = feat.shape[0]
            self.neg = np.vstack((self.neg, feat))
            self.num_neg_added += num
        if self.num_neg_added > self.retrain_limit or force:
            self.num_neg_added = 0
            new_w_b, pos_scores, neg_scores = self.train()
            # scores = np.dot(self.neg, new_w_b[0].T) + new_w_b[1]
            # easy_inds = np.where(neg_scores < self.evict_thresh)[0]
            not_easy_inds = np.where(neg_scores >= self.evict_thresh)[0]
            if len(not_easy_inds) > 0:
                self.neg = self.neg[not_easy_inds, :]
                # self.neg = np.delete(self.neg, easy_inds)
            print('    Pruning easy negatives')
            print('    Cache holds {} pos examples and {} neg examples'.
                  format(self.pos.shape[0], self.neg.shape[0]))
            print('    {} pos support vectors'.format((pos_scores <= 1).sum()))
            print('    {} neg support vectors'.format((neg_scores >= -1).sum()))
            return new_w_b
        else:
            return None

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train SVMs (old skool)')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='models/VGG16_cub/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='output/default/cub_train/{:s}/vgg16_fast_rcnn_iter_40000.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='experiments/cfgs/svm.yml', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='cub_train', type=str)
    parser.add_argument('--partid', dest='partid',
                        help='partid (0: bbox, 1: head, 2: body)',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # pull out features (tricky!)
    cfg.DEDUP_BOXES = 0

    # Must turn this on because we use the test im_detect() method to harvest
    # hard negatives
    cfg.TEST.SVM = True
    
    
    args = parse_args()

    print('Called with args:')
    print(args)
    
    if args.cfg_file is not None:
        cfg_file = osp.join(_init_paths.main_dir, args.cfg_file)
        cfg_from_file(cfg_file)
        
    print('Using config:')
    pprint.pprint(cfg)
    
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    
    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    partid = args.partid   # 0: bbox, 1: head, 2: body
    
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    print 'Part: ', imdb.parts[partid]
    
    
    
    #caffemodel = '/DB/rhome/zxu/workspace/fast-rcnn/fast-rcnn/output/default/cub_train/' +\
    #    imdb.parts[partid-1] + '/vgg16_fast_rcnn_iter_40000.caffemodel'
    caffemodel = osp.join(_init_paths.main_dir, args.caffemodel.format(imdb.parts[partid]))
    prototxt = osp.join(_init_paths.main_dir, args.prototxt)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    #net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    out = os.path.splitext(os.path.basename(caffemodel))[0] + '_svm'
    out_dir = os.path.dirname(caffemodel)


    # enhance roidb to contain flipped examples
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_roidb()
        print 'done'

    SVMTrainer(net, imdb, partid).train()

    filename = '{}/{}.caffemodel'.format(out_dir, out)
    net.save(filename)
    print 'Wrote svm model to: {:s}'.format(filename)
