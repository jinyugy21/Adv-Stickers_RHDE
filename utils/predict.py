from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy
import torch
import torchvision.models as models
import os
import cv2
from models import *
import warnings
import shutil

from utils import rotate
from utils import stick
from utils import mapping3d
from utils import feature
warnings.filterwarnings("ignore")

""" perturb the image """
def perturb_image(xs, backimg, sticker,opstickercv,magnification, zstore, searchspace, facemask):
    xs = np.array(xs)
    #print('xs = ',xs)
    #print('imgs=',image_arr.shape)
    d = xs.ndim
    if(d==1):
        xs = np.array([xs])
    w,h = backimg.size
    
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        #print('making {}-th perturbed image'.format(i),end='\r')
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        rt_sticker = rotate.rotate_bound_white_bg(opstickercv, angle)
        nsticker,_ = mapping3d.deformation3d(sticker,rt_sticker,magnification,zstore,x,y)
        outImage = stick.make_stick2(backimg=backimg, sticker=nsticker, x=x, y=y, factor=xs[i][1])
        imgs.append(outImage)
        
        check_result = int(check_valid(w, h, nsticker, x, y, facemask))
        valid.append(check_result)
            
    return imgs,valid

def check_valid(w, h, sticker, x, y, facemask):
    _,basemap = stick.make_basemap(width=w, height=h, sticker=sticker, x=x, y=y)
    area = np.sum(basemap)
    overlap = facemask * basemap
    retain = np.sum(overlap)
    if(abs(area - retain) > 15):
        return 0
    else:
        return 1

def simple_perturb(xs, backimg, sticker, searchspace, facemask):
    xs = np.array(xs)
    d = xs.ndim
    if(d==1):
        xs = np.array([xs])
    w,h = backimg.size
    
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        print('making {}-th perturbed image'.format(i),end='\r')
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        stickercv = rotate.img_to_cv(sticker)
        rt_sticker = rotate.rotate_bound_white_bg(stickercv, angle)
        outImage = stick.make_stick2(backimg=backimg, sticker=rt_sticker, x=x, y=y, factor=xs[i][1])
        imgs.append(outImage)
        
        check_result = int(check_valid(w, h, rt_sticker, x, y, facemask))
        valid.append(check_result)
    
    return imgs,valid

""" query the model for image's classification """
"""
def predict_type_xxx(image_perturbed, cleancrop):
    typess = [[top-1,...,top-5],...]
    percent = [[probability vector],...]
    return typess,percent
"""
def predict_type_facenet(image_perturbed, cleancrop):
    #print('shape = ',image_perturbed.shape)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('Running on device: {}'.format(device))
    def collate_fn(x):
        return x
    loader = DataLoader(
        image_perturbed,
        batch_size=42,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = torch.load('./models/facenet/net_13_022.pth',map_location='cuda:0').to(device)
    resnet.eval()
    resnet.classify = True
    
    percent = []
    typess = []
    
    for X in loader:
        C = mtcnn(X)   # return tensor list
        C = [cleancrop if x is None else x for x in C]
        batch_t = torch.stack(C)
        #print(batch_t.shape)
        batch_t = batch_t.to(device)
        out = resnet(batch_t).cpu()
        #print('logits\' len = ',len(out[0]))
        #print('true label = ',true_label)
        with torch.no_grad():
            _, indices = torch.sort(out.detach(), descending=True)
            percentage = torch.nn.functional.softmax(out.detach(), dim=1) * 100
            
            for i in range(len(out)):
                cla = [indices[i][0].item(),indices[i][1].item(),indices[i][2].item(),\
                       indices[i][3].item(),indices[i][4].item()]
                typess.append(cla)
                tage = percentage[i]
                percent.append(tage)

    return typess,percent

"""
def initial_predict_xxx(image_perturbed):
    typess = [[top-1,...,top-5],...]
    percent = [[probability vector],...]
    C = mtcnn(image_perturbed,save_path='./test.jpg')
    return typess,percent,C[0]
"""
def initial_predict_facenet(image_perturbed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = torch.load('./models/facenet/net_13_022.pth',map_location='cuda:0').to(device)
    resnet.eval()
    resnet.classify = True
    
    percent = []
    typess = []
    
    C = mtcnn(image_perturbed,save_path='./test.jpg')   # return tensor list
    batch_t = torch.stack(C)
    #print(batch_t.shape)
    batch_t = batch_t.to(device)
    out = resnet(batch_t).cpu()
    with torch.no_grad():
        _, indices = torch.sort(out.detach(), descending=True)
        percentage = torch.nn.functional.softmax(out.detach(), dim=1) * 100
        cla = [indices[0][0].item(),indices[0][1].item(),indices[0][2].item(),\
               indices[0][3].item(),indices[0][4].item()]
        typess.append(cla)
        tage = percentage[0]
        percent.append(tage)

    return typess,percent,C[0]


