import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import re
import os
import random

cap = cv2.VideoCapture(".../room_practice.mp4")

# hsv color range
lower = np.array([15,7,140])
upper = np.array([50,22,184])

# kernel for image dilation
kernel = np.ones((3,3),np.uint8)

change_score = []
prv_old = 0
prv = 0
current = 0

while(True):
    ret, f = cap.read()
    f = f.transpose(1,0,2)
    f = cv2.resize(f, (176, 320), interpolation = cv2.INTER_AREA)
    f = cv2.GaussianBlur(f,(15,15),0)

    hsv = cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask,kernel,iterations = 2)

    mask[160:,:] = 0
    mask[:,:60] = 0
    mask[:,150:] = 0
    mask[:75,:] = 0

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)    
    max_area = cv2.contourArea(contours[0])
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (cv2.contourArea(c) <= max_area) & (cv2.contourArea(c) < 200) & (cv2.contourArea(c) > 74):
            current = x
            max_area = cv2.contourArea(c)
    

    cnt+= 1

    if (current < prv) & (prv > prv_old):
        change_score.append(1)
    else:
        change_score.append(0)

    prv_old = prv
    prv = current

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

font = cv2.FONT_HERSHEY_SIMPLEX

cnt = 0
score = 0
cap = cv2.VideoCapture(path+"room_practice.mp4")

while(True):
    ret, f = cap.read()
    f = f.transpose(1,0,2)
    f = cv2.resize(f, (176, 320), interpolation = cv2.INTER_AREA)
    
    if change_score[cnt] == 1:
        score+= 1

    blur = cv2.GaussianBlur(f,(15,15),0)
    dmy= cv2.putText(frame.copy(),str(score),(70,135), font, 2, (50,50,110), 12, cv2.LINE_AA)
    dmy = cv2.rectangle(dmy,(60,75), (150, 160),color = [50, 50, 110], thickness = 2)

    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(f, f, mask=mask)
    res[175:,:,:] = 0
    res[:,:50,:] = 0
    res[:,150:,:] = 0
    res[:75,:,:] = 0

    kernel = np.ones((3,3),np.uint8)
    res = cv2.dilate(res,kernel,iterations = 3)

    dmy[res==0] = 1
    #dmy[dmy!=102] = 1
    dmy[(dmy!=110)&(dmy!=50)&(dmy!=50)] = 1
    f[dmy!=1] = 1
    nf = f*cv2.cvtColor(dmy, cv2.COLOR_RGB2BGR)

    #cv2.imshow('frame', f)   
    cv2.imwrite('.../frames_2/'+str(cnt)+'.png',nf)
    cnt+= 1

    '''     
    if cnt == 2:
        break
    '''

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


# output path to save the video
pathIn= '.../frames_2/'

# output path to save the video
pathOut = '.../p_trainer_v3.mp4'

# specify frames per second
fps = 33.0

from os.path import isfile, join

# get file names of the frames
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key=lambda f: int(re.sub('\D', '', f)))

frame_list = []

for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_list.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_list)):
    # writing to a image array
    out.write(frame_list[i])

out.release()