# from inspect import currentframe
# from turtle import distance
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from facenet_pytorch import MTCNN, InceptionResnetV1
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
import numpy as np
import glob
import cv2
import os
from numpy import dot
from numpy.linalg import norm
from PIL import Image
import time
#from model_difination import *
from torch2trt import torch2trt
from torch2trt import TRTModule


normalize = transforms.Normalize(mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformID = transforms.Compose([  transforms.ToTensor(), 
                                    normalize ])




#======= hyperparameters & data loaders =======#
cfg = configurations[1]

SEED = cfg['SEED'] # random seed for reproduce results
torch.manual_seed(SEED)

DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

INPUT_SIZE = cfg['INPUT_SIZE']
RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
RGB_STD = cfg['RGB_STD']
EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
BATCH_SIZE = cfg['BATCH_SIZE']
DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
LR = cfg['LR'] # initial LR
NUM_EPOCH = cfg['NUM_EPOCH']
WEIGHT_DECAY = cfg['WEIGHT_DECAY']
MOMENTUM = cfg['MOMENTUM']
STAGES = cfg['STAGES'] # epoch stages to decay learning rate

DEVICE = cfg['DEVICE']
MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
GPU_ID = cfg['GPU_ID'] # specify your GPU ids
PIN_MEMORY = cfg['PIN_MEMORY']
NUM_WORKERS = cfg['NUM_WORKERS']
print("=" * 60)
print("Overall Configurations:")
print(cfg)
print("=" * 60)


train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
    #transforms.Resize([112,112]), # smaller side resized
    #transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = RGB_MEAN,
                            std = RGB_STD),
])

train_transform_in = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
    #transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
    transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = RGB_MEAN,
                            std = RGB_STD),
])

BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                    'ResNet_101': ResNet_101(INPUT_SIZE), 
                    'ResNet_152': ResNet_152(INPUT_SIZE),
                    'IR_50': IR_50(INPUT_SIZE), 
                    'IR_101': IR_101(INPUT_SIZE), 
                    'IR_152': IR_152(INPUT_SIZE),
                    'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                    'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                    'IR_SE_152': IR_SE_152(INPUT_SIZE)}
BACKBONE = BACKBONE_DICT[BACKBONE_NAME]

#BACKBONE_RESUME_ROOT = "D:\\faceRecon\\train_face_recog\\fadata_model\\Backbone_IR_SE_50_Epoch_20_Batch_16360_Time_2022-08-05-01-08_checkpoint.pth"
BACKBONE_RESUME_ROOT = "Backbone_IR_SE_50_Epoch_21_Batch_17178_Time_2022-08-05-01-20_checkpoint.pth"

BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))

BACKBONE.eval().cuda()

x = torch.ones((1, 3, 112, 112)).cuda()
start = time.time()
fNo = 1000
# for i in range (0,fNo):
   
#     pred = BACKBONE(x)

# print("FPS (Without TRT):", int(fNo/(time.time()-start)))

opt_model = "recognition_trt_fp16.pth"
model_trt = torch2trt(BACKBONE, [x], fp16_mode=True, max_workspace_size=1<<25)
torch.save(model_trt.state_dict(), opt_model)
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(opt_model))

start = time.time()
for i in range (0,fNo):
    
    pred = model_trt(x)

print("FPS (With TRT):", int(fNo/(time.time()-start)))