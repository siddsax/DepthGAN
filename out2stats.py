import time
import cv2
import numpy as np
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from test import test
import pdb
import os
import subprocess
from train import train
from os import listdir
from os.path import isfile, join
import sys

opt = TrainOptions().parse()
dFiles1 = [f for f in listdir(opt.dataroot + '/seq/D1') if isfile(join(opt.dataroot + '/seq/D1', f))]
dFiles2 = [f for f in listdir(opt.dataroot + '/seq/D2') if isfile(join(opt.dataroot + '/seq/D2', f))]
dFilesR = [f for f in listdir(opt.dataroot + '/seq/DR') if isfile(join(opt.dataroot + '/seq/DR', f))]
dFiles1.sort()
dFiles2.sort()
dFilesR.sort()
numFiles = min(len(dFiles2), len(dFiles1))
home = opt.dataroot
models = {}
model1 = create_model(opt)

avg = np.zeros((1,5))
for i in range(numFiles):
    print('==========' + str(i) + '========== ' + home + '/seq/D2/' +  dFiles2[i] + '\n')
    f2 = cv2.imread(home + '/seq/D2/' +  dFiles2[i])
    fR = cv2.imread(home + '/seq/DR/' +  dFilesR[i])
    losses, names = model1.findCustomLosses(f2, fR)
    toPrint = ""
    avg += losses
    for k in range(len(losses)):
        toPrint += names[k] + " " + str(losses[k]) + " "
    print(toPrint)

print(avg/numFiles)