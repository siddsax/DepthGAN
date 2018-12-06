import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--f', dest='folder', default='', type=str, help='an integer for the accumulator')
parser.add_argument('--x1', dest='x1', default=73, type=int, help='an integer for the accumulator')
parser.add_argument('--x2', dest='x2', default=529, type=int, help='an integer for the accumulator')
parser.add_argument('--y1', dest='y1', default=259, type=int, help='an integer for the accumulator')
parser.add_argument('--y2', dest='y2', default=815, type=int, help='an integer for the accumulator')
parser.add_argument('--st', dest='st', default=60, type=int, help='an integer for the accumulator')
parser.add_argument('--end', dest='end', default=125, type=int, help='an integer for the accumulator')
args = parser.parse_args()

if len(args.folder) == 0:
  print("Error, folder not given")
  exit()

# This file takes depth img and rgb img, joining them together

depthFiles = [f for f in listdir(args.folder + '/Depth') if isfile(join(args.folder + '/Depth', f))]
rgbFiles = [f for f in listdir(args.folder + '/RGB') if isfile(join(args.folder + '/RGB', f))]

depthFiles.sort()
rgbFiles.sort()

#print(args.st)
depthFiles = depthFiles[args.st:args.end+1]
rgbFiles = rgbFiles[args.st:args.end+1]
numFiles = min(len(rgbFiles), len(depthFiles))
#print(numFiles)
targetFolder = args.folder + '_out'
if not os.path.exists(targetFolder):
    os.makedirs(targetFolder)

for i in range(numFiles):
  print(depthFiles[i])
  fd = cv2.imread(args.folder + '/Depth/' + depthFiles[i])[args.x1:args.x2, args.y1:args.y2]
  fr = cv2.imread(args.folder + '/RGB/' +  rgbFiles[i])[args.x1:args.x2, args.y1:args.y2]

  c = np.concatenate((fr, fd), axis=1)
  cv2.imwrite(targetFolder + '/img_' + str(i) + '.png', c)


