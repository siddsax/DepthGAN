#!/usr/bin/env python
# Master's Thesis - Depth Estimation by Convolutional Neural Networks
# Jan Ivanecky; xivane00@stud.fit.vutbr.cz

import numpy as np
import sys
import PIL
from PIL import Image
import os.path 
import random
from random import randint
import os
#try:
#	os.mkdir('train_colors0')
#except OSError:
#	print('output folder already exists')
#try:
#	os.mkdir('train_depths0')
#except OSError:
#	print('output folder already exists')

counter = 1
for file in os.listdir("train_colors"):
    if file.endswith(".png"):
	print(file)	
	depthFile = file.replace('rgb','depth')
	filePath = 'train_colors/' + file
	depthFilePath = 'train_depths/' + depthFile
	if(os.path.exists(depthFilePath)):
		continue
	print(str(counter) + filePath + ' ' + depthFilePath)
	counter += 1	

	colorOriginal = Image.open(filePath)
	depthOriginal = Image.open(depthFilePath)

	width, height = 561, 427
 	newWidth, newHeight = 420, 320
	borderX = (width - newWidth) / 2
	borderY = (height - newHeight) / 2
	
	try:
		colorNew = colorOriginal.crop((borderX, borderY, width - borderX, height - borderY))
	except:
		continue
	try:
		depthNew = depthOriginal.crop((borderX, borderY, width - borderX, height - borderY))
	except:
		continue
	colorNew.save('train_colors0/' + file)
	depthNew.save('train_depths0/' + depthFile)
		
		


	
