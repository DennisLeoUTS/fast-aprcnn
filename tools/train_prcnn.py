#!/usr/bin/env python

# --------------------------------------------------------
# Fast Part-based R-CNN
# Written by Zhe Xu
# --------------------------------------------------------

import _init_paths
import argparse
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from prcnn.prcnn import PRCNN_classifier
import pprint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train SVMs (old skool)')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='/DB/rhome/zxu/workspace/fast-rcnn/fast-rcnn/models/VGG16_cub/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='/DB/rhome/zxu/workspace/fast-rcnn/fast-rcnn/output/default/cub_train/{:s}/vgg16_fast_rcnn_iter_40000_svm.caffemodel', type=str)
    parser.add_argument('--detpath', dest='detpath',
                        help='path for detection results',
                        default='../output/svm/{:s}/vgg16_fast_rcnn_iter_40000_svm/detections.pkl', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
                        default='../experiments/cfgs/prcnn.yml', type=str)
    parser.add_argument('--imdb_train', dest='imdb_train',
                        help='dataset to train on',
                        default='cub_train', type=str)
    parser.add_argument('--imdb_test', dest='imdb_test',
                        help='dataset to test on',
                        default='cub_test', type=str)
    parser.add_argument('--imdb_weak', dest='imdb_weak',
                        help='web dataset',
                        default='cub_weak', type=str)
    parser.add_argument('--C', dest='C', help='C for training SVM',
                        default=0.001, type=float)
    parser.add_argument('--mode', dest='mode', help='testing mode',
                        default='all', type=str)
    parser.add_argument('--trainmode', dest='trainmode', help='training mode, (denoise,classifyonly)',
                        default='denoise', type=str)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print 'Init main'
    args = parse_args()

    print('Called with args:')
    print(args)
    
    cfg_from_file(args.cfg_file)
    
    print('Using config:')
    pprint.pprint(cfg)
    
    
    
    prcnncls = PRCNN_classifier(args.imdb_train, args.imdb_test, args.imdb_weak, \
                                args.caffemodel, args.detpath, args.prototxt, args.mode, args.trainmode, args.C)
    
    if cfg.PRCNN.PHRASE == 'train':
        '''
        Train SVMs on CUB dataset and test on the testing set
        '''
        prcnncls.load_cub()
        prcnncls.load_cub_test()
        prcnncls.train_svm()
        prcnncls.test_svm()
        
        '''
        Test on the weak set and denoise
        '''
        prcnncls.load_cub_weak()
        prcnncls.test_svm_on_weak()
        prcnncls.noise_removal()
    else:
        '''
        Train SVMs on the augmented dataset
        '''
        prcnncls.load_cub_weak()
        prcnncls.load_cub_test()
        if cfg.PRCNN.WITH_ORIGINAL:
            prcnncls.load_cub()
        prcnncls.merge_train_weak()
        #prcnncls.train_svm()
        prcnncls.test_svm()
        