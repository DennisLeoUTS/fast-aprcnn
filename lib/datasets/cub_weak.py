# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.cub_weak
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

def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))


class cub_weak(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'cub_' + image_set)
        self._image_set = image_set
        #self._part_name = part_name
        #self._devkit_path = self._get_default_path() if devkit_path is None \
        #                    else devkit_path
        self._data_path = cfg.PRCNN.DATA_ADD_DIR
        self._image_path = os.path.join(self._data_path, 'images')
        self._tr_ims_path = os.path.join(self._data_path, 'tr_ims.txt')
        self._nim_per_class = 100

        classes = os.listdir(self._image_path)
        classes = sorted(classes, key=lambda x: int(x[:3]))
        classes.insert(0, '__background__') # always index 0
        self._classes = classes

        self._parts = ['bbox', 'head', 'body']
        
        self.valid_ind = None

        #self._classes = ('__background__', # always index 0
        #                'head')
        #                 self._part_name)

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._part_to_ind = dict(zip(self.parts, xrange(self.num_parts)))
        
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

    def _load_image_set_index_v0(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        
        image_index = []
        image_labels = []
        
        for i in xrange(1,self.num_classes):
            clpath = os.path.join(self._image_path, self._classes[i])
            #ims = os.listdir(clpath)
            ims = sorted_ls(clpath)
            image_index.extend([os.path.join(self._classes[i], ims[j]) for j in range(min(len(ims),self._nim_per_class))])
            image_labels.extend([i for j in range(self._nim_per_class)])
            
        self._image_labels = image_labels
        return image_index
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        fd = open(self._tr_ims_path, 'r')
        l = fd.readlines()
        fd.close()
        
        image_index = []
        image_labels = []
        
        for i in range(len(l)):
            li = l[i].strip('\n')
            lis = li.split(' ')
            image_index.append(lis[1])
            image_labels.append(int(lis[0]))
            
        self._image_labels = image_labels
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        self.gt_list = []
        
        det_DIR = cfg.PRCNN.WEAK_DIR
        for i in range(self.num_parts):
            cache_file = os.path.join(det_DIR, self.parts[i]+'_detections.pkl')
            with open(cache_file, 'rb') as fid:
                all_boxes  = cPickle.load(fid)
                num_images = len(all_boxes[1])
                out_boxes = []
                for j in xrange(num_images):
                    if len(all_boxes[1][j])>0:
                        out_boxes.append(all_boxes[1][j][0])
                    else:
                        out_boxes.append(np.array([-1,-1,-1,-1,-1]))
                self.gt_list.append( out_boxes )
        
        image_index = self.image_index
        print len(self.gt_list[0])
        
        gt_roidb = [self._load_annotation_helper(i)
                    for i in range(len(image_index))]
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #with open(cache_file, 'wb') as fid:
        #    cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    
    
    def _load_annotation_helper(self, i):
        # print 'Loading: {}'.format(filename)
        #def get_data_from_tag(node, tag):
        #    return node.getElementsByTagName(tag)[0].childNodes[0].data

        #with open(filename) as f:
        #    data = minidom.parseString(f.read())

        #objs = data.getElementsByTagName('object')
        gt_box = [self.gt_list[j][i] for j in range(self.num_parts)]
        num_objs = self.num_parts

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_parts), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for iix in range(3):
            if gt_box[iix][0] == -1:
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



    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        
        if cfg.PRCNN.PHRASE == 'train':       # only for testing
            cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')
        elif cfg.PRCNN.PHRASE == 'retrain':
            cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb_retrain.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        
        if cfg.PRCNN.PHRASE == 'train':       # only for testing
            print 'Load roidb only as testing'
            roidb = self._load_selective_search_roidb(None)
        else:                                 # use weak as training either
            print 'Load roidb for training on the augmented dataset'
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
            
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
            #if self.valid_ind != None and self.valid_ind[i]==False:
            #    continue
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
            
        print 'Len ss: ', len(box_list)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    
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

    def evaluate_detections(self, all_boxes, partid, output_dir):
        self._write_cub_results_file(all_boxes, partid, output_dir)
        #self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.cub_weak('weak')
    #print d.name
    #print len(d._image_index)
    #print d.image_path_at(1)
    res = d.gt_roidb()
    #print res[0]
    #from IPython import embed; embed()
    #d._load_cub_annotation()
    #print d._load_annotation_helper(1)
