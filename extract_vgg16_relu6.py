import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DD
import torchnet as tnt
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models
from torchvision import transforms as T

import os
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from PIL import Image
from glob import glob

################################
# load pretrained vgg16
# limit the classifier to first Dense
################################
vgg = models.vgg16(pretrained=True)
vgg_clr = list(vgg.classifier.children())
new_clr = nn.Sequential(*([vgg_clr[0]]))
vgg.classifier = new_clr
print vgg

################################
# Transforms for each class1 image
# 1. FiveCrop      2. ToTensor
# 3. Normalize     4. Stack
################################
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
prep = T.Compose([ T.ToTensor(), normalize ])
transform = T.Compose([
    T.FiveCrop((224,224)),
    T.Lambda(lambda crops: torch.stack([prep(crop) for crop in crops])) 
]) 

################################
# For each class1 video,
#    For each frame of this video,
#      Transform the image using above 
#      For each 5 cropped image,
#         extract vgg16 features
#      take mean of these 5 
#      for each frame, append feat to list
# savemat for each video
################################
# folder names for class1 videos
c1_vid_folders = glob('../UCF101_release/images_class1/*')
output_folder = '../UCF101_release/vgg16_relu6/'
for vid in c1_vid_folders:
    frames = glob(vid+'/*.jpg')
    vgg_all = []
    for img_name in frames:
        print img_name
        img = Image.open(img_name)
        img_crops = transform(img)
        ncrops, c, h, w = img_crops.size()
        five_cropped_img_features = []
        for cropped_img in img_crops:
            feat_i = vgg(cropped_img.unsqueeze(0))
            five_cropped_img_features.append(feat_i)
        five = torch.stack(five_cropped_img_features)
        mean_of_five_crops = five.mean(0).data
        vgg_all.append(mean_of_five_crops)
    vgg_for_vid = torch.stack(vgg_all)
    dic = {}
    fe=vgg_for_vid.numpy()
    fe=fe.reshape((25, 4096))
    dic['Feature'] = fe
    print vid, fe.shape , '==========================='
    savemat(output_folder+vid.split('/')[-1]+'.mat', dic)

