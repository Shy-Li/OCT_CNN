#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:36:18 2022

@author: whitaker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
copyright: Quing Zhu, Optical and Ultrasound Imaging Laboratory
Email: zhu.q@wustl.edu

This code is for plotting predicted probabilities for each B-scan OCT image.
'''
import os, glob
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFont
from PIL import ImageDraw 

image_size = (128,256) # input size for the CNN
nstart = 50 # starting pixel
nsum = 512 # only use 512 pixles in the z-direction
split = 8 # split into 8 sub-images
pix = 125
thresh = 0.7 # exclude sub-images with sum of pixels < 0.7 * whole image
best_model = 103 # best model epoch 
model = keras.models.load_model('models/models_train_on_both2/save_at_' + str(best_model) + '.h5')

isFile = os.path.isdir('exvivo/AUCtest') 
if isFile == False:
    os.mkdir('exvivo/AUCtest')
test_folders = glob.glob('exvivo/patient*')
for ifolder in [4,6]:#range(len(test_folders)): # this is patient number
    folder = test_folders[ifolder]
    print(folder)
    savename = 'exvivo/AUCtest/' + folder[7:] +'_test.csv'  
    # use the threshold to exclude low-intensity images       
    isFile = os.path.isdir(folder+'_both2_test') 
    if isFile == False:
        os.mkdir(folder+'_both2_test')
    else:
        print('folder exists')
   
    pngLst = [i for i in os.listdir(folder) if os.path.isfile(os.path.join(folder,i)) and 'score' not in i]
    score_all= pd.DataFrame(index=range(split),columns= pngLst)
    
    # testing scores for being normal
    for i in range(len(pngLst)):
        scores = []
        imgPath = folder + '/' + pngLst[i] 
        img = np.asarray(Image.open(imgPath))
        img = img[nstart:nsum+nstart,:]
        thresh2 = thresh*np.sum(img)/8
        for ii in range(split):
            imgi = img[:,ii*pix:(ii+1)*pix]
            if np.sum(imgi) <= thresh2:
                score = np.nan
                scores.append(score)
            else:
                imgi = Image.fromarray(imgi).resize(image_size)
                img_array = keras.preprocessing.image.img_to_array(imgi)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model.predict(img_array)
                score = predictions[0][0]
                scores.append(score)
        if ~np.isnan(scores).all():
            score_all.iloc[:,i]= pd.Series(scores)
            
            # plot scores directly on the images to better visualize
            implt =  Image.fromarray(img)
            draw = ImageDraw.Draw(implt)
            fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)    
            for ii in range(split):
                if np.isnan(scores[ii]):
                    t = '*'
                else:
                    t = str(round(scores[ii],4))
                draw.text((ii*pix+round(pix/5), 480), t ,font=fnt, fill = 255)
                draw.line([(ii*pix,0), (ii*pix,512)],fill = 255, width=2)
            implt.save(folder+'_both2_test/'+ pngLst[i][:-4]+'_score.png')
    # save all scores
    score_all.dropna(axis=1, how='all').to_csv(savename) # remove all na columns and save

    
  
