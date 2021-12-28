from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def img_to_cv(image):
    imgarray = np.array(image)
    r,g,b,a = cv2.split(imgarray)
    cvarray = cv2.merge([b, g, r, a])
    return cvarray


def rotate_bound_white_bg(imagecv, angle):

    # imagecv = img_to_cv(image)
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = imagecv.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    
    # # if change the size
    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
    # # compute the new bounding dimensions of the image
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # return cv2.warpAffine(imagecv, M, (nW, nH))
    rotated = cv2.warpAffine(imagecv, M, (w, h),borderValue=(255,255,255,0))
    b, g, r, a = cv2.split(rotated)
    rotated_array = cv2.merge([r, g, b, a])
    rt_sticker = Image.fromarray(np.uint8(rotated_array))
    return rt_sticker

# stickerpic = Image.open('./stickers/bs14_2.png')
# stickercv = img_to_cv(stickerpic)
# r = rotate_bound_white_bg(stickercv, 120)
# r.show()
