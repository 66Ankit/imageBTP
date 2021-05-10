import matplotlib.pyplot as plt
from skimage import data
from skimage import exposure
import numpy as np
from skimage.transform import match_histograms
# from skimage.exposure import match_histogram
import cv2
import os



image = cv2.imread('imgs_out/man.jpg')
hist_image=cv2.calcHist(image,[0],None,[256],[0,256])

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))




def ref():
    Ref=""
    min = 100000000.0
    path="ref"
    for i in os.listdir(path):
        full_path=os.path.join(path,i)
        temp=cv2.imread(full_path)
        hist=cv2.calcHist(temp,[0],None,[256],[0,256])
        print(type(hist))
        value=kl_divergence(hist_image, hist)
        print(value)
        if(value<min):
            min=value
            Ref=i
    return Ref
Ref=ref()

print(Ref)
reference = cv2.imread(os.path.join('ref',Ref))
import numpy as np


matched = match_histograms(image, reference,multichannel=True)
print(matched.shape)
cv2.imshow('orignal',cv2.resize(image,(500,500)))
cv2.imshow('matched',cv2.resize(np.uint8(matched),(500,500)))
cv2.waitKey(0)