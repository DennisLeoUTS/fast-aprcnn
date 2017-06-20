# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.cub
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

class cub(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'cub_' + image_set)
        self._image_set = image_set
        #self._part_name = part_name
        #self._devkit_path = self._get_default_path() if devkit_path is None \
        #                    else devkit_path
        self._data_path = cfg.PRCNN.DATA_DIR

        class_file = os.path.join(self._data_path, 'classes.txt')
        assert os.path.exists(class_file), \
                'Path does not exist: {}'.format(class_file)
        with open(class_file) as f:
            classes = [x.strip().split(' ')[1] for x in f.readlines()]
        classes.insert(0, '__background__') # always index 0
        self._classes = classes

        self._parts = ['bbox', 'head', 'body']

        #self._classes = ('__background__', # always index 0
        #                'head')
        #                 self._part_name)

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._part_to_ind = dict(zip(self.parts, xrange(self.num_parts)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        #assert os.path.exists(self._devkit_path), \
        #        'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        #assert os.path.exists(self._data_path), \
        #        'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images',
                                  index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        train_path_file = os.path.join(self._data_path, 'train_test_split.txt')
        assert os.path.exists(train_path_file), \
                'Path does not exist: {}'.format(train_path_file)
        with open(train_path_file) as f:
            ttsplit = [x.strip().split(' ')[1] for x in f.readlines()]
            
            
        image_set_file = os.path.join(self._data_path, 'images.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split(' ')[1] for x in f.readlines()]
            
        image_label_file = os.path.join(self._data_path, 'image_class_labels.txt')
        assert os.path.exists(image_label_file), \
                'Path does not exist: {}'.format(image_label_file)
        with open(image_label_file) as f:
            image_labels = [int(x.strip().split(' ')[1]) for x in f.readlines()]
        
        if self._image_set=='train': 
            la = '1'
        else:
            la = '0'
        image_labels = [image_labels[i] for i in range(len(image_labels)) if ttsplit[i] == la]
        image_index = [image_index[i] for i in range(len(image_index)) if ttsplit[i] == la]
        self._image_labels = image_labels
        return image_index


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        self._load_cub_annotation()
        gt_roidb = [self._load_annotation_helper(i)
                    for i in range(len(self.image_index))]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set == 'train':
            gt_roidb = self.gt_roidb()
            print 'Len gt: ', len(gt_roidb)
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
        print 'Len ss: ', len(box_list)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_cub_annotation(self):
        """
        Load image and bounding boxes (bbox, head, body) info from mat file in the CUB format.
        """
        filename = os.path.join(self._data_path, 'bird_' + self._image_set + '.mat')
        raw_data = sio.loadmat(filename)
        gt_list = []
        nim = len(raw_data['data'][0])
        #print raw_data['data'][0][3]
        for i in xrange(nim):
            tmp = raw_data['data'][0][i]
            gt = [tmp[x+1][0] for x in range(3)]
            if self.name.endswith('train'):
                tmp = gt[2]
                gt = gt[0:2]
                gt.insert(0, tmp)
            gt_list.append(gt)
        
        self.gt_list = gt_list

    def _load_annotation_helper(self, i):

        # print 'Loading: {}'.format(filename)
        #def get_data_from_tag(node, tag):
        #    return node.getElementsByTagName(tag)[0].childNodes[0].data

        #with open(filename) as f:
        #    data = minidom.parseString(f.read())

        #objs = data.getElementsByTagName('object')
        gt_box = self.gt_list[i]
        num_objs = [1 for j in range(3) if gt_box[j][0] != -1]
        num_objs = sum(num_objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_parts), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for iix in range(3):
            if gt_box[iix][0] == -1:
                #print i, iix, '-1'
                continue
            cls = iix
            # Make pixel indexes 0-based
            x1 = gt_box[ix][0] - 1
            y1 = gt_box[ix][1] - 1
            x2 = gt_box[ix][2] - 1
            y2 = gt_box[ix][3] - 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls+1
            overlaps[ix, cls] = 1.0
            ix += 1

        # overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}


    def _write_cub_results_file(self, all_boxes, partid=0, output_dir='output'):
        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        cls = self.parts[partid]
        cls_ind = 1
        
        filename = os.path.join(output_dir, self.parts[partid] + '_detection_results.txt')
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == [] or dets.shape[0] == 0:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, -1, -1, -1, -1, -1))
                    continue
                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))
                    

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _do_eval(self, all_boxes, partid=0):
        cls = self.parts[partid]
        cls_ind = 1
        gt_roidb = self.gt_roidb()
        
        suc_det = 0
        both_no_det = 0
        
        for im_ind, _ in enumerate(self.image_index):
            dets = all_boxes[cls_ind][im_ind]
            gt_inds = np.where(gt_roidb[im_ind]['gt_classes'] == partid+1)[0]
            gt_boxes = gt_roidb[im_ind]['boxes'][gt_inds, :]
            
            have_dets = len(dets) != 0
            have_gts = len(gt_boxes) != 0
            
            if have_dets and have_gts:
                boxes = dets[0:1, :-1] 
                overlaps = bbox_overlaps(boxes.astype(np.float),
                            gt_boxes.astype(np.float))
                overlap = overlaps[0][0]
                
                # print im_ind, overlap
                if overlap>0.5:
                    suc_det += 1
                    
            elif not have_dets and not have_gts:  # both do not have results
                suc_det += 1 
                both_no_det += 1
                
        acc = float(suc_det)/len(self.image_index)
        
        print 'Detection accuracy for part {:s} is {:.4f} ({:d}/{:d})'\
        .format(self.parts[partid], acc, suc_det, len(self.image_index))
        print '{:d} of them both don\'t have detection results'.format(both_no_det)

    def evaluate_detections(self, all_boxes, partid, output_dir):
        self._write_cub_results_file(all_boxes, partid, output_dir)
        self._do_eval(all_boxes, partid)
        
        

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.cub('train')
    print d.name
    gt_roidb = d.gt_roidb()
    print gt_roidb[0]
    
    #res = d.roidb
    #from IPython import embed; embed()
    #d._load_cub_annotation()
    #print d._load_annotation_helper(1)
