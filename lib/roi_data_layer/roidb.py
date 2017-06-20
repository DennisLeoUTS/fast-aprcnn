# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg,get_output_dir
import utils.cython_bbox
import os
import cPickle
from datasets.factory import get_imdb

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    
    if imdb.name == 'cub_train' or imdb.name == 'cub_test':
        prepare_roidb_for_cub(imdb)
        return 
    elif imdb.name == 'cub_weak' or imdb.name.startswith('nabirds'):
        prepare_roidb_for_weak(imdb)
        return
    
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        try:
            gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        except:
            gt_overlaps = roidb[i]['gt_overlaps']
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        # print max_overlaps, max_classes
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
        
        
        
        
def prepare_roidb_for_cub(imdb):
    print 'Preparing roidb for {}'.format(imdb.name)
    cache_file = os.path.join(imdb.cache_path,
                    imdb.name + '_roidb_all.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb_all = cPickle.load(fid)
        print '{} roidb all loaded from {}'.format(imdb.name, cache_file)
            
        imdb.set_roidb(roidb_all)
        return
    
    roidb = imdb.roidb
    labels = imdb.labels
    
    roidb_all = [[] for _ in range(3)]
    
    tool_add_roidb(imdb, roidb, labels, roidb_all)
    
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb_all, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote roidb all to {}'.format(cache_file)
    
    imdb.set_roidb(roidb_all)
    
    print 'done'
    
    
def prepare_roidb_for_weak(imdb):
    print 'Preparing roidb for {}...'.format(imdb.name)
    roidb = imdb.roidb
    labels = imdb.labels
    
    roidb_all = [[] for _ in range(3)]
    
    if cfg.PRCNN.NOISE_REMOVE:
        cache_file = os.path.join(cfg.PRCNN.WEAK_DIR, 'selected_samples.pkl')
        with open(cache_file, 'rb') as f:
            selected_inds_all = cPickle.load(f)
            tool_add_roidb_weak(imdb, roidb, labels, roidb_all, selected_inds_all)
    else:
        tool_add_roidb(imdb, roidb, labels, roidb_all)
    
    if cfg.PRCNN.WITH_ORIGINAL:
        # get training imdb
        imdb_train = get_imdb('cub_train')
        roidb_train = imdb_train.roidb
        labels_train = imdb_train.labels
        
        tool_add_roidb(imdb_train, roidb_train, labels_train, roidb_all)
    
    imdb.set_roidb(roidb_all)
    
    print 'done'
    

def tool_add_roidb(imdb, roidb, labels, roidb_all): 
    num_images = len(imdb.image_index)
    for i in xrange(num_images):
        imname = imdb.image_path_at(i)
        
        boxes = roidb[i]['boxes']
        flipped = roidb[i]['flipped']
        for k in xrange(3):
            if sum(roidb[i]['gt_classes']==k+1)==0:
                continue
            tmp_rois = {}
            tmp_rois['image'] = imname
            #tmp = roidb[i]['gt_overlaps'].toarray()
            #tmp_rois['gt_overlaps'] = tmp[:,k]
            tmp_rois['gt_overlaps'] = roidb[i]['gt_overlaps'][:,k]
            tmp_rois['boxes'] = boxes
            tmp_rois['flipped'] = flipped
            
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'][:,k]
            
            # max overlap with gt over classes (columns)
            max_classes = np.zeros((len(gt_overlaps)), dtype=np.int)
            # gt class that had the max overlap
            max_overlaps = np.zeros((len(gt_overlaps)), dtype=np.float)
            
            for j in range(len(gt_overlaps)):
                max_overlaps[j] = gt_overlaps[j]
                if gt_overlaps[j]<1e-5:
                    max_classes[j] = 0
                else:
                    max_classes[j] = labels[i]
            
            tmp_rois['max_classes'] = max_classes
            tmp_rois['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 1e-5)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 1e-5)[0]
            assert all(max_classes[nonzero_inds] != 0)
            
            roidb_all[k].append(tmp_rois)
            
            
def tool_add_roidb_weak(imdb, roidb, labels, roidb_all, selected_inds_all): 
    num_images = len(imdb.image_index)
    # print 'num images:', num_images
    for i in xrange(num_images):
        imname = imdb.image_path_at(i)
        
        boxes = roidb[i]['boxes']
        flipped = roidb[i]['flipped']
        for k in xrange(3):
            if selected_inds_all[k].count(i)==0:
                continue
            tmp_rois = {}
            tmp_rois['image'] = imname
            #tmp = roidb[i]['gt_overlaps'].toarray()
            #tmp_rois['gt_overlaps'] = tmp[:,k]
            tmp_rois['gt_overlaps'] = roidb[i]['gt_overlaps'][:,k]
            tmp_rois['boxes'] = boxes
            tmp_rois['flipped'] = flipped
            
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = roidb[i]['gt_overlaps'][:,k]
            
            # max overlap with gt over classes (columns)
            max_classes = np.zeros((len(gt_overlaps)), dtype=np.int)
            # gt class that had the max overlap
            max_overlaps = np.zeros((len(gt_overlaps)), dtype=np.float)
            
            for j in range(len(gt_overlaps)):
                max_overlaps[j] = gt_overlaps[j]
                if gt_overlaps[j]<1e-5:
                    max_classes[j] = 0
                else:
                    max_classes[j] = labels[i]
            
            tmp_rois['max_classes'] = max_classes
            tmp_rois['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 1e-5)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 1e-5)[0]
            assert all(max_classes[nonzero_inds] != 0)
            
            roidb_all[k].append(tmp_rois)

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'
    num_images = len(roidb)
    print 'NUM images:', num_images
    # Infer number of classes from the number of columns in gt_overlaps
    sh = roidb[0]['gt_overlaps'].shape
    
    '''
    !!!to be modified 
    '''
    if len(sh)==1:
        num_classes = 556 # for cub dataset, turn the output into 201 classes
    else:
        num_classes = sh[1]
    
    
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = \
                _compute_targets(rois, max_overlaps, max_classes)
        
    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        if targets == None:
            continue
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
                
    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2) + cfg.EPS
                
    # Normalize targets
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        if targets == None:
            continue
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
    
    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    #print means.shape
    #print means
    return means.ravel(), stds.ravel()

def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(rois[ex_inds, :],
                                                     rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    try:
        gt_assignment = ex_gt_overlaps.argmax(axis=1)
    except:
        return None
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)

    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1] = targets_dx
    targets[ex_inds, 2] = targets_dy
    targets[ex_inds, 3] = targets_dw
    targets[ex_inds, 4] = targets_dh
    return targets
