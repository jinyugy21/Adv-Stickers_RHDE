# _*_ coding:utf-8 _*_
import numpy as np
import dlib
from cv2 import cv2
from PIL import Image
from utils import stick
from torchvision import datasets

def face_landmarks(initial_pic):
    dotsets = np.zeros((1,81,2))
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')
    
    pic_array = np.array(initial_pic)
    r,g,b = cv2.split(pic_array)
    img = cv2.merge([b, g, r])
    #img = cv2.imread(pic_dir)                          

    imgsize = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    rects = detector(img_gray, 1)                      
    #print('num of rects=',len(rects),rects[1])
    #print(len(rects))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    #print(landmarks)
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])           
        #print(idx,pos)
        if(idx >= 0 and idx <= 67):
            dotsets[0][idx] = pos
        elif(idx == 78):
            dotsets[0][68] = pos
        elif(idx == 74):
            dotsets[0][69] = pos
        elif(idx == 79):
            dotsets[0][70] = pos
        elif(idx == 73):
            dotsets[0][71] = pos
        elif(idx == 72):
            dotsets[0][72] = pos
        elif(idx == 80):
            dotsets[0][73] = pos
        elif(idx == 71):
            dotsets[0][74] = pos
        elif(idx == 70):
            dotsets[0][75] = pos
        elif(idx == 69):
            dotsets[0][76] = pos
        elif(idx == 68):
            dotsets[0][77] = pos
        elif(idx == 76):
            dotsets[0][78] = pos
        elif(idx == 75):
            dotsets[0][79] = pos
        elif(idx == 77):
            dotsets[0][80] = pos

    return dotsets,imgsize

def circle_mark(facemask,dot,brw):
    dot = dot.astype(np.int16)
    dotlen = len(dot)
    for i in range(dotlen):
        x1,y1 = dot[i]
        facemask[x1,y1] = brw
        if(i == dotlen-1):
            j = 0
        else:
            j = i+1
        x2,y2 = dot[j]
        if(y2 - y1 != 0):
            k = (x2 - x1) / (y2 - y1)
            symbol = 1 if(y2 - y1 > 0) else -1
            for t in range(symbol*(y2 - y1)-1):
                y3 = y1 + symbol * (t + 1)
                x3 = int(round(k * (y3 - y1) + x1))
                # print('x1,y1,x2,y2',x1,y1,x2,y2)
                # print('x3,y3 = ',x3,y3)
                facemask[x3,y3] = brw

    dot = np.array(dot)
    lower = np.min(dot,axis = 0)[1]
    upper = np.max(dot,axis = 0)[1]
    for h in range(lower,upper+1):
        left = 0
        right = 0
        cruitl = np.min(dot,axis = 0)[0]
        cruitr = np.max(dot,axis = 0)[0]
        for i in range(cruitl-1,cruitr+2):
            if(facemask[i][h] == brw):
                left = i
                break
        for j in reversed(list(range(cruitl-1,cruitr+2))):
            if(facemask[j][h] == brw):
                right = j
                break
        left_cursor = left
        right_cursor = right
        # print('h = ',h)
        # print('left_cursor,right_cursor = ',left_cursor,right_cursor)
        if(left_cursor != right_cursor):        
            while True:
                facemask[left_cursor][h] = brw
                left_cursor = left_cursor + 1
                if(facemask[left_cursor][h] == brw):
                    break
            while True:
                facemask[right_cursor][h] = brw
                right_cursor = right_cursor - 1
                if(facemask[right_cursor][h] == brw):
                    break
    return facemask

def make_mask(initial_pic):
    dotsets,imgsize = face_landmarks(initial_pic)
    facemask = np.zeros((imgsize[1],imgsize[0]))
    #----------face--------------
    face = dotsets[0][:17]
    face2 = dotsets[0][68:]
    face = np.vstack((face,face2))
    #print(face)
    facemask = circle_mark(facemask,face,brw=1)

    #---------eyebrow-----------
    browl = dotsets[0][17:22]
    browr = dotsets[0][22:27]
    facemask = circle_mark(facemask,browl,brw=0)
    facemask = circle_mark(facemask,browr,brw=0)

    #----------eye--------------
    eyel = dotsets[0][36:42]
    eyer = dotsets[0][42:48]
    facemask = circle_mark(facemask,eyel,brw=0)
    facemask = circle_mark(facemask,eyer,brw=0)

    #---------mouth-------------
    mouth = dotsets[0][48:61]
    facemask = circle_mark(facemask,mouth,brw=0)

    #---------nose--------------
    #nose = np.vstack((dotsets[0][31:36],dotsets[0][42],dotsets[0][27],dotsets[0][39]))
    nose = np.vstack((dotsets[0][31:36],dotsets[0][29]))
    right = [dotsets[0][27][0]+1,dotsets[0][27][1]]
    left = [dotsets[0][27][0]-1,dotsets[0][27][1]]
    nose = np.vstack((dotsets[0][31:36],right,left))
    facemask = circle_mark(facemask,nose,brw=0)

    facemask = facemask.transpose()

    #facemask[5][15]=1
    # cv2.imshow("outImg",facemask)
    # #cv2.imshow("outImg",facemask)
    # cv2.waitKey(0)
    # num_space = np.sum(facemask).astype(int)
    # print(num_space)
    
    return facemask


def count_face(initial_pic):
    dotsets = np.zeros((1,81,2))
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')
    
    pic_array = np.array(initial_pic)
    r,g,b = cv2.split(pic_array)
    img = cv2.merge([b, g, r])
    #img = cv2.imread(pic_dir)                          

    imgsize = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    rects = detector(img_gray, 1)                      
    num = len(rects)
    #print('num of rects=',len(rects),rects[1])
    #print(len(rects))
    return num

