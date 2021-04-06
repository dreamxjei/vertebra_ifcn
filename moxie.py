import numpy as np
import random
import os, sys

from PIL import Image


mdir = '/home-2/jwei9@jhu.edu/work2/mamba'  # master directory
sdir = os.path.join(mdir, 'scoliosis')  # scoliosis directory
ddir = os.path.join(sdir, 'ifcn', 'data')  # data directory

idir = os.path.join(ddir, 'train', '0')

# load vertebra mask
xray = Image.open(os.path.join(idir, 'image_phi=82_theta=98.tiff'))
xray = np.array(xray)
print('xray shape is:',np.shape(xray))  # 1280 x 640

vertebrae = []

files = os.listdir(idir); files.sort()

for idx in files:
    if idx.startswith('image'):
        continue  # skip image
    ivert = int(idx.split('_')[1])
    vertebrae.append(ivert)

print(len(vertebrae),'vertebrae found:',vertebrae)
    
mask = np.zeros((np.amax(vertebrae)+1, 1280, 640))  # mask size is 0 to largest vertebrae
for idx in files:
    if idx.startswith('image'):
        continue  # skip image
    ivert = int(idx.split('_')[1])
    vertebrae.append(ivert)

    raw = Image.open(os.path.join(idir, idx))
    mask[ivert, :, :] = np.array(raw)

verts = vertebrae  # make it easier to follow github code
chosen_vert = verts[random.randint(1, len(verts) - 1)]  # picks a random vertebra
# print('chosen vert is:',chosen_vert)

# create corresponding instance memory and ground truth