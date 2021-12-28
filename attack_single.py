from PIL import Image 
from PIL import ImageEnhance
import numpy as np
import scipy
import torch
import torchvision.models as models
from torchvision import datasets, transforms
import os
import xlwt
from tqdm import tqdm
import time 
import copy
import random
import argparse
import cv2

from heuristicsDE import differential_evolution
from utils import Config
from utils import rotate
from utils import stick
from utils import predict
from utils import mapping3d
from utils import feature
from utils import tools

""" Perturb the image with the given individual(xs) and get the prediction of the model """
def predict_classes(cleancrop, xs, gorb, initial_pic, target_class, searchspace, \
    sticker, opstickercv, magnification, zstore, facemask, minimize=True):
    imgs_perturbed, valid = predict.perturb_image(xs, initial_pic, \
        sticker, opstickercv, magnification, zstore, searchspace, facemask)
    predictions = []
    le = len(imgs_perturbed)
    
    rank, pred_p = eval('predict.predict_type_{}(imgs_perturbed,cleancrop)'.format(threat_model))
     
    global timess    
    timess=timess+1
    global convert, start, latter
    global generate_rank, generate_score, best_rank, best_score
    print('times = ',timess,'start = ',start,'convert = ',convert)
    for i in range(le):
        if(rank[i][0] != target_class):   # untarget
            probab = -1 * pinf
        else:
            label2 = rank[i][1]
            probab1 = pred_p[i][target_class].item()
            if(start == False):
                probab2 = pred_p[i][label2].item()
                a,b = 1,0
                probab = a * probab1 - b * probab2
            elif(start == True):
                probab2 = pred_p[i][latter].item()
                #a,b = 0.3,0.7
                #probab = a * probab1 - b * probab2
                beta = 20 if threat_model!='arcface' else 1
                probab = beta*(probab1 - probab2)/probab1 + (probab1 - probab2)
        if(valid[i] == 0):
            probab = pinf

        predictions.append(np.array(probab))
    predictions = np.array(predictions)
    duplicate = copy.deepcopy(predictions)
    current_optimal = int(duplicate.argsort()[0])
    mingap = pred_p[current_optimal][rank[current_optimal][0]].item() - pred_p[current_optimal][rank[current_optimal][1]].item()
    #print('mingap = ',mingap)
    if(gorb == 0):
        generate_rank.append([rank[current_optimal][0],rank[current_optimal][1]])
        generate_score.append([pred_p[current_optimal][rank[current_optimal][0]].item(),pred_p[current_optimal][rank[current_optimal][1]].item()])
        sid = int(xs[current_optimal][0])
        #print('x,y = ',int(searchspace[sid][0]),int(searchspace[sid][1]))
    elif(gorb == 1):
        best_rank.append([rank[current_optimal][0],rank[current_optimal][1]])
        best_score.append([pred_p[current_optimal][rank[current_optimal][0]].item(),pred_p[current_optimal][rank[current_optimal][1]].item()])
        sid = int(xs[current_optimal][0])
        #print('x,y = ',int(searchspace[sid][0]),int(searchspace[sid][1]))
    if(start==False and rank[current_optimal][0] == target_class and mingap <= bound):
        start = True
        latter = rank[current_optimal][1]
        convert = True
        print('--------------convert to target attack--------')
        #print('mingap = ',mingap)
    return predictions, rank, convert, pred_p, valid

def convert_energy(rank, pred_p, valid, target_class):
    global convert
    convert = False
    print('----------convert_energy------------')
    predictions = []
    for i in range(len(rank)):
        if(rank[i][0] != target_class):   # untarget
            probab = -1 * pinf
        else:
            label2 = rank[i][1]
            probab1 = pred_p[i][target_class].item()
            probab2 = pred_p[i][latter].item()
            #a,b = 0.3,0.7
            #probab = a * probab1 - b * probab2
            beta = 20 if threat_model!='arcface' else 1
            probab = beta*(probab1 - probab2)/probab1 + (probab1 - probab2)
        if(valid[i] == 0):
            probab = pinf
        predictions.append(np.array(probab))
    predictions = np.array(predictions)
    return predictions    

def single_predict(cleancrop,xs, initial_pic, true_label, searchspace, \
    sticker,opstickercv,magnification, zstore, facemask):
    # imgs_perturbed, valid = predict.perturb_image(xs, initial_pic, \
    #     sticker, opstickercv, magnification, zstore, searchspace, facemask)
    imgs_perturbed, valid = predict.simple_perturb(xs, initial_pic, \
        sticker, searchspace, facemask)
    
    rank, pred_p = eval('predict.predict_type_{}(imgs_perturbed,cleancrop)'.format(threat_model))
    predictions = []
    for i in range(len(imgs_perturbed)):
        if(rank[i][0] != true_label):   # untarget
            probab = -1 * pinf
        else:
            probab = pred_p[i][true_label].item()

        if(valid[i] == 0):
            probab = pinf
        predictions.append(probab)
    predictions = np.array(predictions)
    return predictions

"""  If the prediction is what we want (misclassification or targeted classification), return True """
def attack_success(cleancrop,x, initial_pic, target_class, searchspace, \
    sticker,opstickercv,magnification, zstore, facemask, targeted_attack=False):
    attack_image, valid = predict.perturb_image(x, initial_pic, \
        sticker, opstickercv, magnification, zstore, searchspace, facemask)
    
    rank, _ = eval('predict.predict_type_{}(attack_image,cleancrop)'.format(threat_model))
    predicted_class = rank[0][0]
    #print('callback: predicted_class=',predicted_class,'valid[0]=',valid[0],x)
    if ((targeted_attack and predicted_class == target_class and valid[0]==1) or
        (not targeted_attack and predicted_class != target_class and valid[0]==1)):
        return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function

def region_produce(cleancrop,xs, true_label, searchspace, pack_searchspace, trace_searchspace, initial_pic, \
    sticker, opstickercv, magnification, zstore, facemask):
    h, w = int(facemask.shape[0]), int(facemask.shape[1])
    len_relative = len(xs)
    len_per = np.zeros((len_relative,1))  # the number of valid dots around the current dot
    pots = []                             # The whole set of perturbation vectors considered in inbreeding
    inbreeding = []
    for i in list(range(len_relative)):   # for each individual
        cur = int(xs[i][0])
        alp = xs[i][1]
        angle = xs[i][2]
        x = int(searchspace[cur][0])
        y = int(searchspace[cur][1])
        neighbors = tools.adjacent_coordinates(x,y,s=1)
        temp = 0
        for j in range(len(neighbors)):
            p = tools.num_clip(0,w-1,int(neighbors[j][0]))
            q = tools.num_clip(0,h-1,int(neighbors[j][1]))
            if(alp in trace_searchspace[q][p]):              # if this dot has been visited
                judge = random.random()
                if(judge <= 0.5):                            # change the step
                    slide = 2
                    while(1):
                        #print('change step')
                        far_neighbors = tools.adjacent_coordinates(x,y,s=slide)
                        pn = int(far_neighbors[j][0])
                        qn = int(far_neighbors[j][1])
                        if(alp in trace_searchspace[qn][pn]):
                            slide = slide + 1
                        else:
                            break
                    trace_searchspace[qn][pn].append(alp)
                    attribute = pack_searchspace[qn][pn]
                    if(attribute >= 0):
                        temp = temp + 1
                        pots.append([attribute,alp,angle])
                else:                                        # change the alpha using random
                    #print('change alpha')
                    attribute = pack_searchspace[q][p]
                    alp_ex = random.uniform(0.8,0.98)
                    if(attribute >= 0):
                        temp = temp + 1
                        pots.append([attribute,alp_ex,angle])
                    trace_searchspace[q][p].append(alp_ex)
            else:
                trace_searchspace[q][p].append(alp)
                attribute = pack_searchspace[q][p]
                #print('attribute = ',attribute)
                if(attribute >= 0):
                    temp = temp + 1
                    pots.append([attribute,alp,angle])
        len_per[i][0] = temp
    predictions = single_predict(cleancrop,pots, initial_pic, true_label, searchspace, \
        sticker,opstickercv,magnification, zstore, facemask)
    cursor = 0
    #print('len_per = ',len_per.T)
    for i in range(len_relative):
        sublen = len_per[i][0]
        if(sublen != 0):
            upper = int(cursor + sublen)
            subset = predictions[int(cursor):upper]
            better = np.argsort(subset)[0]
            inbreeding.append(pots[int(cursor+better)])
        else:
            inbreeding.append(xs[i])
        cursor = cursor + sublen
    
    #print('len_relative, inbreeding = ',len_relative, len(inbreeding))
    return inbreeding

def attack(idx,true_label,initial_pic,sticker,opstickercv,magnification,\
    cleancrop,zstore,target=None, maxiter=30, popsize=40):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else true_label

    facemask = feature.make_mask(initial_pic) # valid=1, unvalid=0

    num_space = np.sum(facemask).astype(int)
    searchspace = np.zeros((num_space,2))         # store the coordinate(Image style)
    pack_searchspace = copy.deepcopy(facemask)-2  # record the id, unvalid=-2
    trace_searchspace = []                        # mark whether it has been accessed
    for i in range(facemask.shape[0]):
        col = [[-1] for j in range(facemask.shape[1])]
        trace_searchspace.append(col)
    k = 0
    for i in range(facemask.shape[0]):
        for j in range(facemask.shape[1]):
            if(facemask[i][j] == 1):
                searchspace[k] = (j,i)
                # pack_searchspace[i][j] = k
                k = k + 1
    np.random.shuffle(searchspace)
    for i in range(len(searchspace)):
        x = int(searchspace[i][0])
        y = int(searchspace[i][1])
        pack_searchspace[y][x] = int(i)
    bounds = [(0,num_space), (0.8,0.98),(0,359)]
    print('---------begin attack---------------')
    
    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs,gorb):
        return predict_classes(cleancrop,xs, gorb, initial_pic, target_class, searchspace, \
            sticker,opstickercv,magnification, zstore, facemask, target is None)
    
    def callback_fn(x, convergence):
        return attack_success(cleancrop,x, initial_pic, target_class, searchspace, \
            sticker,opstickercv,magnification, zstore, facemask, targeted_attack)
    
    def region_fn(xs):
        return region_produce(cleancrop,xs, true_label, searchspace, pack_searchspace, trace_searchspace, \
            initial_pic, sticker,opstickercv,magnification, zstore, facemask)
    
    def ct_energy(ranks, pred_ps, valids):
        return convert_energy(ranks, pred_ps, valids, target_class)
    # Differential Evolution
    attack_result = differential_evolution(
        predict_fn, region_fn, ct_energy, bounds, maxiter=maxiter, popsize=popsize,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image, valid = predict.perturb_image(attack_result.x, initial_pic, \
        sticker, opstickercv, magnification, zstore, searchspace, facemask)
    
    rank, pred_p = eval('predict.predict_type_{}([initial_pic],cleancrop)'.format(threat_model))
    rank2, pred_p2 = eval('predict.predict_type_{}(attack_image,cleancrop)'.format(threat_model))
    attack_image[0].save('./results_img/{}.png'.format(idx))
    
    prior_probs = pred_p[0][target_class].item()
    predicted_class = rank2[0][0]
    predicted_probs = pred_p2[0][predicted_class].item()
    d1 = [rank[0][0],rank[0][1]]
    score1 = [pred_p[0][rank[0][0]].item(),pred_p[0][rank[0][1]].item()]
    d2 = [rank2[0][0],rank2[0][1]]
    score2 = [pred_p2[0][rank2[0][0]].item(),pred_p2[0][rank2[0][1]].item()]
    
    actual_class = true_label
    success = (predicted_class != actual_class ) and valid[0]==1
    cdiff = pred_p[0][actual_class].item() - pred_p2[0][actual_class].item()

    sid = int(attack_result.x[0])
    x = int(searchspace[sid][0])
    y = int(searchspace[sid][1])
    factor = attack_result.x[1]
    angle = attack_result.x[2]
    vector = [x, y, factor, angle, sid, attack_result.x[0]]

    return [actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, vector,d1,score1,d2,score2]


if __name__=="__main__":
    
    model_set = ['arcface', 'facenet', 'sphereface','cosface']
    pinf, ninf = 99.9999999, 0.0000001
    convert = False          # indicate whether DE needs to re-compute energies to compare with target result
    start = False            # whether start target attack from untarget style
    latter = 0               # target class
    generate_rank, generate_score, best_rank, best_score = [],[],[],[]
    timess = 0               # Record the query times based on batch
    
    opt = Config()
    threat_model = model_set[opt.id_threat]
    initial_pic = opt.pic
    true_label = opt.gtlabel
    bound = opt.bound
    idx = opt.idx
    
    "--- sticker processing ---"
    stickerpath = './stickers/{}.png'.format(opt.sticker_name)
    stickerpic = Image.open(stickerpath)
    scale1 = stickerpic.size[0]//23
    scale2 = opt.scale
    magnification = scale2/scale1
    operate_sticker = stick.change_sticker(stickerpic,scale1)
    sticker = stick.change_sticker(stickerpic,scale2)
    opstickercv = rotate.img_to_cv(operate_sticker)

    rank, _, cleancrop = eval('predict.initial_predict_{}([initial_pic])'.format(threat_model))
    
    if(rank[0][0] == true_label):
        zstore = mapping3d.generate_zstore(initial_pic)
        r = attack(idx,true_label, initial_pic, sticker,opstickercv,magnification,cleancrop,zstore)
        
        
