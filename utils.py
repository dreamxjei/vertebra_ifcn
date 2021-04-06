import random
import numpy as np
from data_augmentation import elastic_transform, gaussian_blur, gaussian_noise, random_crop


def force_inside_img(x, patch_size, img_shape):
    x_low = int(x - patch_size / 2)
    x_up = int(x + patch_size / 2)
    if x_low < 0:
        x_up -= x_low
        x_low = 0
    elif x_up > img_shape[0]:  # 2d, is [2] in github
        x_low -= (x_up - img_shape[0])
        x_up = img_shape[0]
    return x_low, x_up


def extract_random_patch(img, mask, weight, verts, idx, subset, empty_interval=5, patch_size=128):
    flag_empty = False

    # list available vertebrae
    # verts = np.unique(mask)  # passed from dataloader
    chosen_vert = verts[random.randint(1, len(verts) - 1)]  # avoid 0, background
    print('chosen vert is:',chosen_vert)

    # create corresponde instance memory and ground truth - goes bottom to top
    # ins_memory = np.copy(mask)  # you can basically do this too and set others to 0
    height, width, layers = np.shape(mask)
    ins_memory = np.zeros((height, width, layers))

    for layer in range(layers):  # basically, if vert is above or equal to chosen, set to 0, else 1
        cur_layer = mask[:,:,layer]
        if layer <= chosen_vert:
            continue  # stays 0
        else:
            ins_memory[:,:,layer] = cur_layer  # change to 1
    
    gt = np.zeros((height, width, layers))
    gt[:,:,chosen_vert] = mask[:,:,chosen_vert]  # ground truth is current vertebrae only

    # send empty mask sample in certain frequency
    if idx % empty_interval == 0:  # every 5 iterations, send empty
        patch_center = [np.random.randint(0, s) for s in img.shape]
        x = patch_center[0]  # should be right... x is [0] (height)
        y = patch_center[1]

        # for instance memory
        gt = np.copy(mask)  # will modify later
        flag_empty = True
    else:
        # indices = np.nonzero(mask == chosen_vert)
        indices = np.nonzero(mask[:,:,chosen_vert])
        try:  # DEBUG for empty masks breaking lower/upper
            lower = [np.min(i) for i in indices]
            upper = [np.max(i) for i in indices]
        except ValueError:  # log it and just use the empty case
            print('vertebrae',chosen_vert,'does not have nonzero indices')
            patch_center = [np.random.randint(0, s) for s in img.shape]
            x = patch_center[0]  # should be right... x is [0] (height)
            y = patch_center[1]

        # for instance memory
        gt = np.copy(mask)  # will modify later
        flag_empty = True

        # lower = [np.min(i) for i in indices]
        # upper = [np.max(i) for i in indices]
        # random center of patch
        x = random.randint(lower[0], upper[0])  # should be right... x is [0] (height)
        y = random.randint(lower[1], upper[1])

    # force random patches' range within the image
    print('img.shape is:',img.shape)
    x_low, x_up = force_inside_img(x, patch_size, img.shape)
    y_low, y_up = force_inside_img(y, patch_size, img.shape)

    # crop the patch
    img_patch = img[x_low:x_up, y_low:y_up]
    ins_patch = ins_memory[x_low:x_up, y_low:y_up, :]
    gt_patch = gt[x_low:x_up, y_low:y_up, :]
    weight_patch = weight[x_low:x_up, y_low:y_up, :]

    #  if the label is empty mask
    if flag_empty:
        ins_patch = np.copy(gt_patch)
        gt_patch = np.zeros_like(ins_patch)
        weight_patch = np.ones_like(ins_patch)  # empty patch has no weight preference

    # Random on-the-fly Data Augmentation
    if subset == 'train':
        # 50% chance elastic deformation  # TODO: does not work yet with 26-layered arrays
        # if np.random.rand() > 0.5:
            # img_patch, gt_patch, ins_patch, weight_patch = elastic_transform(img_patch, gt_patch, ins_patch,
                                                                            #  weight_patch, alpha=20, sigma=5)
        # 50% chance gaussian blur
        if np.random.rand() > 0.5:
            img_patch = gaussian_blur(img_patch)
        # 50% chance gaussian noise
        if np.random.rand() > 0.5:
            img_patch = gaussian_noise(img_patch)

        # 50% random crop along z-axis  # meh it's 128x128 anyways.
        # if np.random.rand() > 0.5:
            # img_patch, ins_patch, gt_patch, weight_patch = random_crop(img_patch, ins_patch, gt_patch,
                                                                    #    weight_patch)

    # decide label of completeness(partial or complete)
    vol = np.count_nonzero(gt == 1)
    sample_vol = np.count_nonzero(gt_patch == 1)
    c_label = 0 if float(sample_vol / (vol + 0.0001)) < 0.98 else 1

    # why is this even here
    # img_patch = np.expand_dims(img_patch, axis=0)
    # ins_patch = np.expand_dims(ins_patch, axis=0)
    # gt_patch = np.expand_dims(gt_patch, axis=0)
    # weight_patch = np.expand_dims(weight_patch, axis=0)
    # c_label = np.expand_dims(c_label, axis=0)

    return img_patch, ins_patch, gt_patch, weight_patch, c_label
