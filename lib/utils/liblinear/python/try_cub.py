from liblinearutil import *
import os
import pickle
import numpy as np

model_path = '/DB/rhome/zxu/workspace/fast-rcnn/fast-rcnn/output/svm/cub_test/vgg16_fast_rcnn_iter_40000_svm'

trainFeatureFile = model_path+'/training_features.pkl'
pkl_file = open(trainFeatureFile, 'rb')
trnFeature = pickle.load(pkl_file)
pkl_file.close()

testFeatureFile = model_path+'/testing_features.pkl'
pkl_file = open(testFeatureFile, 'rb')
tetFeature = pickle.load(pkl_file)
pkl_file.close()

lf = '/DB/rhome/zxu/Datasets/CUB_200_2011/image_class_labels.txt'
fd = open(lf)
ll = fd.readlines()
fd.close()
labels = [int(l.strip('\n').split(' ')[1]) for l in ll]

spl = '/DB/rhome/zxu/Datasets/CUB_200_2011/train_test_split.txt'
fd = open(spl)
ll = fd.readlines()
fd.close()
tts = [int(l.strip('\n').split(' ')[1]) for l in ll]

trainLabel = [labels[l] for l in range(len(labels)) if tts[l]==1]
testLabel = [labels[l] for l in range(len(labels)) if tts[l]==0]

prob  = problem(trainLabel, trnFeature.tolist())
param = parameter('-c 1')
m = train(prob, param)

p_label, p_acc, p_val = predict(testLabel, tetFeature.tolist(), m)
save_model('part_train_only.model', m)
