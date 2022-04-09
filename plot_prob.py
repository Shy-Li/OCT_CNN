#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copyright: Quing Zhu, Optical and Ultrasound Imaging Laboratory
Email: zhu.q@wustl.edu

This code is for plotting predicted probabilities for each B-scan OCT image.
"""
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageFile
from PIL import ImageFont
from PIL import ImageDraw 
import pandas as pd
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
image_size = (128,256) # input size for the CNN
nsum = 512 # only use 512 pixles in the z-direction
best_idx = 0
split = 8 # split into 8 sub-images
best_model = 103 # best model epoch #

isFile = os.path.isdir('whole_images2/AUC_both2_2') 
if isFile == False:
    os.mkdir('whole_images2/AUC_both2_2')
   
for k in [10,14]: # this is patient number
    print(k)
    if k <= 4:
        thresh = 1e6 # use the threshold to exclude low-intensity images
    else: 
        thresh = 2.232e6           
    
    if k > 4:
        pix = 279  # after patient #4, there are 2322 pixels in the x-direction            
    else:
        pix = 125 # until patient #4, there are 1000 pixels in the x-direction
        

model = keras.models.load_model('models/models_train_on_both2/save_at_' + str(best_model) + '.h5')
folder = "whole_images2/tumor" + str(k) + '_2' # change this according to the folder of interest
isFile = os.path.isdir(folder+"/both2") 
if isFile == False:
    os.mkdir(folder+"/both2")
pngLst = [i for i in os.listdir(folder) if os.path.isfile(os.path.join(folder,i)) and \
      "split8" in i]
score_all= pd.DataFrame(index=range(split),columns= pngLst )

# testing scores for being normal
for i in range(len(pngLst)):
    scores = []
    imgPath = folder + "/" + pngLst[i] 
    img = np.asarray(Image.open(imgPath))
    for ii in range(split):
        imgi = img[:512,ii*pix:(ii+1)*pix]
        if np.sum(imgi[50:nsum,:]) <= thresh:
            score = np.nan
            scores.append(score)
        else:
            imgi = Image.fromarray(imgi).resize(image_size,Image.BILINEAR)
            img_array = keras.preprocessing.image.img_to_array(imgi)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            score = predictions[0][0]
            #print(score, "number", i + 1)
            scores.append(score)
    if ~np.isnan(scores).all():
        score_all.iloc[:,i]= pd.Series(scores)
    #     implt =  Image.open(imgPath)
    #     draw = ImageDraw.Draw(implt)
    #     fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

        # for ii in range(split):
        #     if np.isnan(scores[ii]):
        #         t = '*'
        #     else:
        #         t = str(round(scores[ii],4))
        #     draw.text((ii*pix+round(pix/5), 350), t ,font=fnt, fill = 255)
        #     draw.line([(ii*pix,0), (ii*pix,512)],fill = 255, width=2)
        # #implt.show();
        # implt.save(folder+"/"+"both2/"+ pngLst[i][:-4]+'_score.png')

score_all.dropna(axis=1, how='all').to_csv('whole_images2/AUC_both2_2/Cancer' +str(k) +"_both2.csv") # remove all na columns and save

# folder = "whole_images2/normal" + str(k) # +'_2'
# isFile = os.path.isdir(folder+"/both2") 
# if isFile == False:
#     os.mkdir(folder+"/both2")
# pngLst = [i for i in os.listdir(folder) if os.path.isfile(os.path.join(folder,i)) and \
#       "split8" in i]
# score_all= pd.DataFrame(index=range(split),columns= pngLst )

# # testing scores for being normal
# for i in range(len(pngLst)):
#     scores = []
#     imgPath = folder + "/" + pngLst[i] 
#     img = np.asarray(Image.open(imgPath))
#     for ii in range(split):
#         imgi = img[:512,ii*pix:(ii+1)*pix]
#         if np.sum(imgi[50:nsum,:]) <= thresh:
#             score = np.nan
#             scores.append(score)
#         else:
#             imgi = Image.fromarray(imgi).resize(image_size,Image.BILINEAR)
#             img_array = keras.preprocessing.image.img_to_array(imgi)
#             img_array = tf.expand_dims(img_array, 0)
#             predictions = model.predict(img_array)
#             score = predictions[0][0]
#             # print(score)
#             scores.append(score)
#     if ~np.isnan(scores).all():
#         score_all.iloc[:,i]= pd.Series(scores)
#         # implt =  Image.open(imgPath)
#         # draw = ImageDraw.Draw(implt)
#         # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

#         # for ii in range(split):
#         #     if np.isnan(scores[ii]):
#         #         t = '*'
#         #     else:
#         #         t = str(round(scores[ii],4))
#         #     draw.text((ii*pix+round(pix/5), 350), t ,font=fnt, fill = 255)
#         #     draw.line([(ii*pix,0), (ii*pix,512)],fill = 255, width=2)
#         # #implt.show();
#         # implt.save(folder+"/"+"both2/"+ pngLst[i][:-4]+'_score.png')

# score_all.dropna(axis=1, how='all').to_csv('whole_images2/AUC_both2_2/Normal'+str(k)+"_both2.csv")
