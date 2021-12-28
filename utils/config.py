import torch
from PIL import Image
from torchvision import datasets

class Config(object):
    bound = 15             # critical value of the gap between label1 and label2
    id_threat = 1          # id of threat model in ['arcface', 'facenet', 'sphereface','cosface']
    sticker_name = 'bs12'
    scale = 12             # The scale of the sticker

    data_dir = './datasets/lfw_images'
    idx = 13050
    dataset = datasets.ImageFolder(data_dir)
    pic = dataset[idx][0]
    gtlabel = dataset[idx][1]
    
    
    # pic = Image.open()
    # idx = -1
    # gtlabel = 

    
