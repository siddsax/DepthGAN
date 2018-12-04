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

if not os.path.exists(opt.dataroot + "/seq"):
    os.makedirs(opt.dataroot + "/seq")
if not os.path.exists(opt.dataroot + "/seq/D1"):
    os.makedirs(opt.dataroot + "/seq/D1")
if not os.path.exists(opt.dataroot + "/seq/D2"):
    os.makedirs(opt.dataroot + "/seq/D2")
if not os.path.exists(opt.dataroot + "/seq/DR"):
    os.makedirs(opt.dataroot + "/seq/DR")

if not os.path.exists(opt.dataroot + "/seq/train"):
    os.makedirs(opt.dataroot + "/seq/train")
if not os.path.exists(opt.dataroot + "/seq/test"):
    os.makedirs(opt.dataroot + "/seq/test")
if not os.path.exists(opt.dataroot + "/seq/result"):
    os.makedirs(opt.dataroot + "/seq/result")

dFiles1 = [f for f in listdir(opt.dataroot + '/seq/D1') if isfile(join(opt.dataroot + '/seq/D1', f))]
dFiles2 = [f for f in listdir(opt.dataroot + '/seq/D2') if isfile(join(opt.dataroot + '/seq/D2', f))]
dFilesR = [f for f in listdir(opt.dataroot + '/seq/DR') if isfile(join(opt.dataroot + '/seq/DR', f))]
dFiles1.sort()
dFiles2.sort()
dFilesR.sort()
numFiles = min(len(dFiles2), len(dFiles1))
home = opt.dataroot
directions = ['AtoB', 'BtoA']
opt.dataroot = home + '/seq'

models = {}
opt.which_direction = ['AtoB']
model1 = create_model(opt)
model1.setup(opt)
models['AtoB'] = model1
opt.which_direction = ['BtoA']
model2 = create_model(opt)
model2.setup(opt)
models['BtoA'] = model2

opt.niter = 100
opt.niter_decay = 100
opt.continue_train=0
file = open("Errors_" + opt.name + ".txt", "w+")
for i in range(numFiles):
    print('==========' + str(i) + '==========\n')
    f1 = cv2.imread(home + '/seq/D1/' + dFiles1[i])
    f2 = cv2.imread(home + '/seq/D2/' +  dFiles2[i])
    fR = cv2.imread(home + '/seq/DR/' +  dFilesR[i])
    cTr = np.concatenate((f1, f2), axis=1)
    cTe = np.concatenate((f2, fR), axis=1)
    cOriginal = cTe
    cv2.imwrite(home + '/seq/train' + '/img.png', cTr)
    cv2.imwrite(home + '/seq/test' + '/img.png', cTe)
    cv2.imwrite(home + '/seq/result_' + opt.name + '/img_0.png', cTe)
    file.write( '==========' + str(i) + '==========\n')
    losses, names = models['AtoB'].findCustomLosses(f2, fR)
    toPrint = ""
    for k in range(len(losses)):
        toPrint += names[k] + " " + str(losses[k]) + " "
    print(toPrint + " " + "({}, {})".format(home + '/seq/D2/' +  dFiles2[i], home + '/seq/DR/' +  dFilesR[i]))
    file.write(toPrint + "\n")
    for direction in directions:
        opt.which_direction = direction
        models[direction].setup(opt)
        models[direction] = train(opt, models[direction])
        opt.which_direction = 'AtoB'
        for j in range(1, 2):
            print("================================")
            losses = test(opt, models[direction], file=file)
            fo = cv2.imread('results/' + opt.name + '/test_latest/images/img_fake_B.png')
            cTe = np.concatenate((fo, fR), axis=1)
            cRes = np.concatenate((f2, fo, fR), axis=1)
            cv2.imwrite(home + '/seq/test' + '/img.png', cTe)
            cv2.imwrite(home + '/seq/result_' + opt.name + '/img_' + str(i) + '_' + direction + '_' + str(j) + '.png', cRes)
        cv2.imwrite(home + '/seq/test' + '/img.png', cOriginal)
        file.write( '--------------------\n')
    opt.niter = 20
    opt.niter_decay = 20
    file.write( '====================\n')
    file.close()
    file = open("Errors_" + opt.name + ".txt", "a+")

#rmse Orgl 0.282412201111 wtO 0.27085609261 wO 0.248778315004
#rel Orgl 0.0706886438254 wtO 0.0629697941125 wO 0.0590944112449
#t1 Orgl 0.960062486221 wtO 0.964832330936 wO 0.972391958085
#t2 Orgl 0.987742521633 wtO 0.993751033399 wO 0.99699788842
#t3 Orgl 0.996850284529 wtO 0.9981571215 wO 0.999776699598

# srun -t 13:58:00 --mem=24G --gres=gpu:1 --constraint='kepler|pascal' python seqGAN.py --dataroot=./classTranslation --name classRoom_l1_100_2 --continue_train=1 --save_epoch_freq=20 --display_freq=1 --print_freq=1 --how_many=1000 --lambda_L1=100.0
