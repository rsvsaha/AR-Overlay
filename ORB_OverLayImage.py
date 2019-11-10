# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:48:39 2019

@author: Rishav
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
substitute_quad=cv2.imread('download.jpg')
def overLayImg(points,original_img):
    #print(points)
    #print(substitute_quad.shape)
    cx=int(points[0][0][0])
    cy=int(points[0][0][1])
    w=points[2][0][0]
    h=points[2][0][1]
    height=int(h-cx)
    width=int(w-cy)
    #print("cx:{} cy:{}".format(cx,cy))
    #print("w:{},h:{}".format(h,w))
    #print((width,height))
    if(width>0 and height>0):
        if(abs(cx-width)<original_img.shape[0] and abs(cy-height)<original_img.shape[1]):
            out_quad=cv2.resize(substitute_quad,(height,width))
            print(out_quad.shape)
            print("cx:{} cy:{}".format(cx,cy))
            #print("w:{},h:{}".format(h,w))
            #print((width,height))
            #print(original_img.shape)
            if(original_img[cx:cx+width,cy:cy+height].shape == out_quad.shape):
                original_img[cx:cx+width,cy:cy+height]= out_quad
            #print(out_quad.shape)
img1=cv2.imread('NavySeal.jpg',0)
#cv2.imshow("WIKI",img1)
orb=cv2.ORB_create()
kp1,des1=orb.detectAndCompute(img1,None)
img_out=cv2.drawKeypoints(img1,kp1,None,color=(0,0,255),flags=0)
cv2.imwrite("Keypoints.jpg",img_out)
h,w=img1.shape
pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
MIN_MATCHES=15
cap=cv2.VideoCapture()
cap.open(0)
while True:

    _,img_rgb=cap.read()
    #img_rgb=cv2.flip(img_rgb,1)
    img2=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
    kp2,des2=orb.detectAndCompute(img2,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=bf.match(des1,des2)
    matches=sorted(matches,key=lambda x: x.distance)
    #print(len(matches))
    if len(matches)>MIN_MATCHES:
        #img_out=cv2.drawMatches(img1,kp1,img2,kp2,matches[:MIN_MATCHES],0,flags=2)
        src_pts=np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        dst=cv2.perspectiveTransform(pts,M)
        img_out=cv2.polylines(img_rgb,[np.int32(dst)],True,(0,0,255),3,cv2.LINE_AA)
        overLayImg(dst,img_out)
    else:
        img_out=img_rgb
        print("NOT ENOUGH MATCHES")
    
    cv2.imshow('MYFRAME',img_out)
    #print(img_out.shape)
    k=cv2.waitKey(30)
    if k==ord('q') & 0xFF:
        #cv2.imwrite('sliced.jpg',sav_img)
        break
    
cap.release()
cv2.destroyAllWindows()
        




