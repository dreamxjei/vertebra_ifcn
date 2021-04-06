import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset
from utils import extract_random_patch
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, op, mask_json_path, transforms=None):
        self.transform = transforms
        self.op = op
        with open(mask_json_path, 'r') as f:
            self.mask = json.load(f)
        self.mask_num = len(self.mask)
        self.mask_value = [value for value in self.mask.values()]
        self.mask_value.sort()
        try:
            if self.op == 'train':
                self.data_dir = os.path.join(input_dir, 'train')
            elif self.op == 'val':
                self.data_dir = os.path.join(input_dir, 'val')
            elif self.op == 'test':
                self.data_dir = os.path.join(input_dir, 'test')
        except ValueError:
            print('op should be either train, val or test!')

    def __len__(self):
        return len(next(os.walk(self.data_dir))[1])

    def __getitem__(self, idx):
        # print('dataloader going through number',idx)
        idir = os.path.join(self.data_dir, str(idx))  # data/train/0 etc
        # print('dataloader looking in idir:',idir)
        
        allfiles = list(os.listdir(idir))
        allfiles.sort()
        
        # get height, width  # MXJ resize
        sample_data = allfiles[0]
        sample_data = Image.open(os.path.join(idir, sample_data))
        
        # rescale to 1280 x 640 / 2, so 640 x 320
        # sample_data = sample_data.resize((320, 640))  # PIL flipped for some reason
        sample_data = np.array(sample_data)  # to numpy
        height, width = np.shape(sample_data)  # 640 x 320
        
        mask_array = np.zeros((height, width, self.mask_num))
        weight_array = np.zeros((height, width, self.mask_num))
        verts = []  # list of unique vertebra

        for i in range(len(allfiles)):
            filename = allfiles[i]
            if filename.startswith('image'):
                img_name = filename
            elif filename.startswith('vertebra'):
                vertebra_num = np.uint8(filename.split('_')[1])
                rawpath = os.path.join(idir, filename)
                raw = Image.open(rawpath)
                # raw = raw.resize((320,640))
                raw = np.array(raw)
                mask_array[:,:,vertebra_num] = raw
                verts.append(vertebra_num)
            elif filename.startswith('weight'):
                vertebra_num = np.uint8(filename.split('_')[2])
                rawpath = os.path.join(idir, filename)
                raw = Image.open(rawpath)
                # raw = raw.resize((320,640))
                raw = np.array(raw)
                weight_array[:,:,vertebra_num] = raw

        img = Image.open(os.path.join(idir, img_name)).convert('RGB')
        # img = img.resize((320,640))
        img = np.array(img)
        mask = mask_array
        weight = weight_array
        
        img = np.float32(img)
        mask = np.float32(mask)
        weight = np.float32(weight)

        # specific to IFCN: extract training patch. gt = ground truth
        subset = self.op
        empty_interval = 5
        patch_size = 128
        img_patch, ins_patch, gt_patch, weight_patch, c_label = extract_random_patch(img,
                                                                                    mask,
                                                                                    weight,
                                                                                    verts,
                                                                                    idx,
                                                                                    subset,
                                                                                    empty_interval,
                                                                                    patch_size
                                                                                    )

        # convert data to float, just in case
        img_patch = np.float32(img_patch)  # 128 x 128 x 3
        ins_patch = np.float32(ins_patch)
        gt_patch = np.float32(gt_patch)
        weight_patch = np.float32(weight_patch)
        c_label = np.float32(c_label)

        # c_label = torch.tensor(c_label)
        # try this is getting c_label to tensor doesn't work
        # c_label = np.expand_dims(c_label, axis=0)
        # c_label = torch.from_numpy(c_label)

        # Transform all image/mask patches, but not c_label
        if self.transform:
            img_patch, ins_patch, gt_patch, weight_patch = self.img_transform(img_patch,
                                                                            ins_patch, 
                                                                            gt_patch,
                                                                            weight_patch
                                                                            )

        # Use dictionary to output
        # sample = {'img': img, 'mask': mask, 'weight': weight}
        sample = {'img': img_patch, 'ins': ins_patch, 'gt': gt_patch, 'weight': weight_patch, 'c_label': c_label}

        return sample

    def img_transform(self, img, ins, gt, weight):
        img = self.transform(img)
        ins = self.transform(ins)
        gt = self.transform(gt)
        weight = self.transform(weight)

        return img, ins, gt, weight
