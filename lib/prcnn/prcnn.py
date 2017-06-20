from datasets.factory import get_imdb
import os
os.environ['GLOG_minloglevel'] = '3'
import caffe
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
import cPickle
from utils.timer import Timer
import numpy as np
import cv2
from liblinearutil import *
import h5py


class PRCNN_classifier(object):
    def __init__(self, imdb_train_name, imdb_test_name, imdb_weak_name, \
                 caffemodelpath, detectionpath, prototxt, mode='all', trainmode='denoise', C=0.001):
        self.imdb_train = get_imdb(imdb_train_name)
        print 'Loaded dataset `{:s}` for training'.format(self.imdb_train.name)
        self.imdb_test = get_imdb(imdb_test_name)
        print 'Loaded dataset `{:s}` for testing'.format(self.imdb_test.name)
        self.imdb_weak = get_imdb(imdb_weak_name)
        print 'Loaded dataset `{:s}` as weakly supervised dataset'.format(self.imdb_weak.name)
        
        self.caffemodelpath = caffemodelpath
        self.detectionpath = detectionpath
        self.prototxt = prototxt
        
        fake_net = fake(os.path.splitext(os.path.basename(caffemodelpath))[0]);
        self.output_dir = get_output_dir(self.imdb_test, fake_net)
        self.weak_dir = get_output_dir(self.imdb_weak, fake_net)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.weak_dir):
            os.makedirs(self.weak_dir)
        
        #self.NUM = 60
        
        self.layer = 'fc7'
        
        self.labels_train = self.imdb_train.labels
        self.labels_test = self.imdb_test.labels
        self.labels_weak = self.imdb_weak.labels
        self.mode = mode
        self.trainmode = trainmode
        self.C = C

    def load_cub(self):
        '''
        Load features on the original CUB dataset based on the caffemodel detectors
        '''
        # for each part
        #out_file = os.path.join(self.output_dir, 'training_features.pkl')
        #if os.path.exists(out_file):
        #    with open(out_file, 'rb') as f:
        #        self.features_train = cPickle.load(f)
        out_file = os.path.join(self.output_dir, 'training_features.h5')
        if os.path.exists(out_file):
            f = h5py.File(out_file,'r')
            self.features_train = f['features'][:]
            f.close()
            return 
        
        #else:
        print 'Start extracting training features...'
        print 'Output file will be wrote into {:s}'.format(out_file)
        part_features_train = [np.zeros((self.imdb_train.num_images, 4096)) for _ in range(self.imdb_train.num_parts)]
        for i in xrange(self.imdb_train.num_parts):
            part_features_train[i] = self.load_feature_part_train(self.imdb_train, i, self.prototxt, self.caffemodelpath)
        self.features_train = self.merge_features(self.imdb_train, part_features_train) 
            
        #with open(out_file, 'wb') as f:
        #    cPickle.dump(self.features_train, f, cPickle.HIGHEST_PROTOCOL)
        f = h5py.File(out_file,'w')
        f.create_dataset('features', data=self.features_train)
        f.close()
        
        
    def load_cub_test(self):
        '''
        Load testing features on the original CUB dataset based on the caffemodel detectors
        '''   
        #out_file = os.path.join(self.output_dir, 'testing_features.pkl')
        #if os.path.exists(out_file):
        #    with open(out_file, 'rb') as f:
        #        self.features_test = cPickle.load(f)
        out_file = os.path.join(self.output_dir, 'testing_features.h5')
        if os.path.exists(out_file):
            f = h5py.File(out_file,'r')
            self.features_test = f['features'][:]
            f.close()
            return 
        
        print 'Start extracting testing features...'
        print 'Output file will be wrote into {:s}'.format(out_file)
        part_features_test = [np.zeros((self.imdb_test.num_images, 4096)) for _ in range(self.imdb_train.num_parts)]
        for i in xrange(self.imdb_test.num_parts):
            #part_features_test[i] = self.load_feature_part_train(self.imdb_test, i, prototxt, caffemodelpath)
            part_features_test[i] = self.load_feature_part_test(self.imdb_test,\
                                i, self.prototxt, self.caffemodelpath, self.detectionpath)
        self.features_test = self.merge_features(self.imdb_test, part_features_test)
        
        f = h5py.File(out_file,'w')
        f.create_dataset('features', data=self.features_test)
        f.close()

        #with open(out_file, 'wb') as f:
        #    cPickle.dump(self.features_test, f, cPickle.HIGHEST_PROTOCOL)
            
            
    def load_cub_weak(self):
        '''
        Load features on the web dataset based on the caffemodel detectors
        '''
        # for each part
        out_file = os.path.join(self.weak_dir, 'features.h5')
        if os.path.exists(out_file):
            f = h5py.File(out_file,'r')
            self.features_weak = f['features'][:]
            f.close()
            return 
        #out_file = os.path.join(self.weak_dir, 'features.pkl')
        #if os.path.exists(out_file):
        #    'Print feature file already exists! Load...'
        #    with open(out_file, 'rb') as f:
        #        self.features_weak = cPickle.load(f)
        #    return 
        
        print 'Start extracting weak features...'
        print 'Output file will be wrote into {:s}'.format(out_file)
        part_features_weak = [np.zeros((self.imdb_weak.num_images, 4096)) for _ in range(self.imdb_weak.num_parts)]
        #part_features_weak = [np.zeros((10, 4096)) for _ in range(self.imdb_weak.num_parts)]
        for i in xrange(self.imdb_weak.num_parts):
            part_features_weak[i] = self.load_feature_part_test(self.imdb_weak, \
                                i, self.prototxt, self.caffemodelpath, self.detectionpath)
        self.features_weak = self.merge_features(self.imdb_weak, part_features_weak)
        
        
        f = h5py.File(out_file,'w')
        f.create_dataset('features', data=self.features_weak)
        f.close()
        
        #with open(out_file, 'wb') as f:
        #    cPickle.dump(self.features_weak[:100, :], f, cPickle.HIGHEST_PROTOCOL)
            
        
    
    
    def load_feature_part_train(self, imdb, partid, prototxt, caffemodelpath):
        '''
        For training samples, use ground truth part bounding boxes to extract features
        '''
        print 'Start loading training features'
        # load caffenet
        partname = imdb.parts[partid]
        caffe.set_mode_gpu()
        caffe.set_device(0)
        caffemodel = caffemodelpath.format(partname)
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(caffemodel))[0]
        
        roidb = imdb.gt_roidb()
        #print roidb[0]

        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}
        
        # compute features
        num_images = len(imdb.image_index)
        #num_images = self.NUM
        feat_part = np.zeros((num_images, 4096))
        
        
        for i in xrange(num_images):
            im = cv2.imread(imdb.image_path_at(i))
            # no detected samples
            #num_objs = len(all_boxes[1][i])
            #print num_objs
            sel_boxes = np.zeros((1, 4), dtype=np.uint16)
            inds = np.where((roidb[i]['gt_classes'] == partid+1))[0]
            if len(inds) == 0:
                feat = np.zeros((1, 4096))
            else:
                sel_boxes[0,:] = roidb[i]['boxes'][inds, :]
                _t['im_detect'].tic()
                scores, boxes = im_detect(net, im, sel_boxes)
                feat = net.blobs[self.layer].data
                _t['im_detect'].toc()
                if i%10 == 0: 
                    print 'Load training features: part {:d}, image {:d}/{:d} in {:.3f}'\
                    .format(partid, i, num_images, _t['im_detect'].average_time)
                
            feat_part[i,:] = feat
        
        print 'Done'
        return feat_part
    
    
    def load_feature_part_test(self, imdb, partid, prototxt, caffemodelpath, detectionpath=None):
        '''
        For testing samples, use predicted part locations to extract features.
        The argument detectionpath is assigned if the detectionpath is not set as default
        '''
        print 'Start loading testing features'
        # load caffenet
        partname = imdb.parts[partid]
        caffe.set_mode_gpu()
        caffe.set_device(0)
        caffemodel = caffemodelpath.format(partname)
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(caffemodel))[0]
        
        # load detection results
        if not detectionpath==None:
            det_res_file = self.detectionpath.format(imdb.name)
        else:
            output_dir = get_output_dir(imdb, net)
            print output_dir
            det_res_file = os.path.join(output_dir, 'detections.pkl')
            
        print 'Load detection results from {:s}'.format(det_res_file)
        assert os.path.exists(det_res_file), 'Detection result file not exists'
        with open(det_res_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        
        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}
        
        # compute features
        num_images = len(imdb.image_index)
        #num_images = self.NUM
        feat_part = np.zeros((num_images, 4096))
        for i in xrange(num_images):
            im = cv2.imread(imdb.image_path_at(i))
            
            # no detected samples
            #num_objs = len(all_boxes[1][i])
            #print num_objs
            
            sel_boxes = np.zeros((1, 4), dtype=np.uint16)
            
            if len(all_boxes[partid][i]) == 0:
                feat = np.zeros((1, 4096))
            else:
                sel_boxes[0,:] = all_boxes[1][i][0][:-1]   # 0 for background, 1 for foreground (part)
                _t['im_detect'].tic()
                scores, boxes = im_detect(net, im, sel_boxes)
                _t['im_detect'].toc()
                feat = net.blobs[self.layer].data
                if i%10 == 0: 
                    print 'Load testing features: part {:d}, image {:d}/{:d}'.format(partid, i, num_images)
            
            feat_part[i,:] = feat
            
        print 'Done'
        return feat_part
    
    
            
    def merge_features(self, imdb, part_features):
        print 'Start merging features'
        num_images = imdb.num_images
        #num_images = self.NUM
        features = np.zeros((num_images, 4096 * imdb.num_parts))
        for i in xrange(num_images):
            for j in xrange(imdb.num_parts):
                features[i, j*4096 : (j+1)*4096] = part_features[j][i,:]
        return features
    
    
    def merge_train_weak(self):
        if not cfg.PRCNN.WITH_ORIGINAL:
            self.labels_train = self.labels_weak
            self.features_train = self.features_weak
            return
        
        out_file = os.path.join(self.output_dir, 'training_features.pkl')
        weak_file = os.path.join(self.weak_dir, 'features.pkl')
        assert os.path.exists(out_file), 'Training features not ready!'
        assert os.path.exists(weak_file), 'Training features not ready!'
        with open(out_file, 'rb') as f:
            features_train = cPickle.load(f)
        with open(weak_file, 'rb') as f:
            features_weak = cPickle.load(f)
        
        nweak = features_weak.shape[0]
        print 'Training samples: #', features_train.shape[0], len(self.labels_train)
        print 'Weak samples: #', nweak, len(self.labels_weak)
            
        if cfg.PRCNN.NOISE_REMOVE:
            cache_file = os.path.join(cfg.PRCNN.WEAK_DIR, 'selected_samples.pkl')
            with open(cache_file, 'rb') as f:
                selected_inds_all = cPickle.load(f)
                print 'Selected_inds_all', len(selected_inds_all[0]), len(selected_inds_all[1]), len(selected_inds_all[2])
            
           
            valid_inds = [i for i in xrange(nweak) if \
                          selected_inds_all[0].count(i)>0 or selected_inds_all[1].count(i)>0 or selected_inds_all[2].count(i)>0]
            print len(valid_inds)
        else:
            valid_inds = [1 for i in xrange(nweak)]
        
        self.features_train = np.vstack((features_train, features_weak[valid_inds, :]))
        self.labels_weak = np.array(self.labels_weak)
        self.labels_train = np.hstack((np.array(self.labels_train), self.labels_weak[valid_inds]))
        print 'After filtering: Training samples: #', self.features_train.shape[0]
        print 'Training labels: #', self.labels_train.shape
        
        
    def scale_feature(self, fea, ppp=0.3):
        fea = np.sign(fea)*np.power(np.abs(fea), ppp)
        return fea
        
    def train_svm(self):
        self.train_svm_ori()
        #pass
        
    def train_svm_ori(self):
        print 'Start training svms'
        if os.path.exists(os.path.join(self.output_dir, 'part_train.model')):
            print '   Model already exists! Load model...'
            self.model = load_model(os.path.join(self.output_dir, 'part_train.model'))
            return
        
        train_fea = self.features_train
        # self.scale_feature(train_fea)
        prob  = problem(self.labels_train, train_fea.tolist())
        
        param = parameter('-c {} -q'.format(self.C))
        self.model = train(prob, param)
        save_model(os.path.join(self.output_dir, 'part_train.model'), self.model)
     
     
    def test_svm(self):
        if self.mode=='part':
            self.test_svm_single_part()
        elif self.mode=='tune':
            self.test_svm_tune_C()
        else:
            self.train_svm_ori()
            self.test_svm_ori()
        
    def test_svm_ori(self): 
    #def test_svm(self):
        print 'Start testing svms'
        #w = self.svm.coef_
        #b = self.svm.intercept_[0]
        test_fea = self.features_test
        #self.scale_feature(test_fea)
        p_label, p_acc, p_val = predict(self.labels_test, test_fea.tolist(), self.model)
        
        acc = p_acc[0]
        #print 'Accuracy=', acc, '%'
        fd = open(os.path.join(self.output_dir, 'result.txt'), 'wt')
        fd.write(str(acc))
        fd.close()
        
        
    def test_svm_single_part(self):
        accs = []
        for k in xrange(3):
            train_fea = self.features_train[:, k*4096:(k+1)*4096]
            test_fea = self.features_test[:, k*4096:(k+1)*4096]
            prob  = problem(self.labels_train, train_fea.tolist())
            print 'Part {}: -c {} -q'.format(self.imdb_train.parts[k], self.C)
            param = parameter('-c {} -q'.format(self.C))
            model = train(prob, param)
            p_label, p_acc, p_val = predict(self.labels_test, test_fea.tolist(), model)
            accs.append(p_acc[0])
        print accs
        
    
    def test_svm_tune_C(self):
    #def test_svm(self):
        print 'Start testing svms'
        train_fea = self.features_train
        test_fea = self.features_test
        #train_fea = self.scale_feature(train_fea)
        #test_fea  = self.scale_feature(test_fea)
        
        clist = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        #clist = list(0.00001*np.array([5,8,10,20,30,50]))
        accs = []
        for C in clist:
            prob  = problem(self.labels_train, train_fea.tolist())
            print '-c {} -q'.format(C)
            param = parameter('-c {} -q'.format(C))
            model = train(prob, param)
            p_label, p_acc, p_val = predict(self.labels_test, test_fea.tolist(), model)
            accs.append(p_acc[0])
        print accs
        
    def test_svm_on_weak(self):
        cache_file = os.path.join(self.weak_dir, 'predict_labels_weak.pkl')
        
        if os.path.exists(cache_file):
            print 'Testing results already exists'
            with open(cache_file, 'rb') as fid:
                p_label = cPickle.load(fid)
                
        else:
            print 'Start testing svms on weak dataset'
            #w = self.svm.coef_
            #b = self.svm.intercept_[0]
            p_label, p_acc, p_val = predict(self.labels_weak, self.features_weak.tolist(), self.model)
            
            acc = p_acc[0]
            #print 'Accuracy=', acc, '%'
            fd = open(os.path.join(self.weak_dir, 'result_weak.txt'), 'wt')
            fd.write(str(acc))
            fd.close()
            
            with open(cache_file, 'wb') as fid:
                cPickle.dump(p_label, fid, cPickle.HIGHEST_PROTOCOL)
            
        self.weak_iscorrect = np.array(p_label)==np.array(self.labels_weak)
        print '{} of {} examples are correctly classified'.format(sum(self.weak_iscorrect), len(self.labels_weak))
        
        
        
    def noise_removal(self):
        '''
        Remove noise according to the classification and localization results
        '''
        cache_file = os.path.join(cfg.PRCNN.WEAK_DIR, 'selected_samples.pkl')
        
        selected_inds_all = []
        for i in xrange(self.imdb_weak.num_parts):
            print 'Part', self.imdb_weak.parts[i] 
            det_res_file = os.path.join(self.weak_dir, self.imdb_weak.parts[i] + '_detections.pkl')
            assert os.path.exists(det_res_file), \
                'detection result file does not exist: {}'.format(det_res_file)
                
            with open(det_res_file, 'rb') as fid:
                all_boxes = cPickle.load(fid)
            
            num_images = len(all_boxes[1])   # 1: for part
            print '#:', num_images
            pos_scores = []
            neg_scores = []
            all_scores = np.ones((num_images, 1)) * (-999)
            for j in xrange(num_images):  
                if len(all_boxes[1][j]) > 0:
                    score = all_boxes[1][j][0][-1]
                    if self.weak_iscorrect[j]:
                        pos_scores.append(score)
                    else:
                        neg_scores.append(score)
                    all_scores[j] = score
            
            print 'Pos scores, ', len(pos_scores), pos_scores[:10]
            print 'Neg scores, ', len(neg_scores), neg_scores[:10]
            mp = np.median(pos_scores)
            mn = np.median(neg_scores)
            print 'Medians: ', mp, mn
            
            if self.trainmode == 'denoise':
                if mp>0:
                    thresh_high = mp/2    # high threshold for wrongly classified examples
                else:
                    thresh_high = mp*2
                if mn>0:
                    thresh_low  = mn/2    # low threshold for correctly classificed examples
                else:
                    thresh_low = mn*2
            else:  # use only correctly classified examples
                thresh_high = 999
                thresh_low = -998
            
            '''
            thresh_high = mn - (mp-mn)*.5
            thresh_low = mn - (mp-mn)*1.5
            '''
                
            print 'Thresh', thresh_high, thresh_low
            
            cls_correct_inds = np.where(self.weak_iscorrect==1)[0]    # correctly classified examples
            cls_wrong_inds = np.where(self.weak_iscorrect==0)[0]
            
            selected_inds = cls_correct_inds.tolist()        # from classification results
            print 'Original cls correct:', len(selected_inds)
            
            
            for k in xrange(len(cls_correct_inds)):           # remove examples with low detection scores
                if all_scores[cls_correct_inds[k]]<thresh_low:  # even if the classification result is correct
                    selected_inds.remove(cls_correct_inds[k])
            print 'After removing low detection results:', len(selected_inds)
            
            for k in xrange(len(cls_wrong_inds)):           # append examples with high detection scores
                if all_scores[cls_wrong_inds[k]]>thresh_high:   # even if the classification result is wrong
                    selected_inds.append(cls_wrong_inds[k])
            print 'After adding high detection results:', len(selected_inds)
            
            
            for s in selected_inds:
                assert len(all_boxes[1][s])>0, 'Problem!!! on part %d, index %s'%(i,s)
            
            selected_inds.sort()
            selected_inds_all.append(selected_inds)
            
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(selected_inds_all, fid, cPickle.HIGHEST_PROTOCOL)
        
        

class fake:
    def __init__(self, name):
        self.name = name


#if __name__ == '__main__':
#    prcnncls = PRCNN_classifier(args.imdb_train, args.imdb_test, args.caffemodel)
#    prcnncls.learn_cub(args.prototxt)
