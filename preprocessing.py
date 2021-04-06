import os
import logging
import numpy as np

from scipy import ndimage
from pathlib import Path
from PIL import Image

# logging = logging.getLogger(__name__)
logging.basicConfig(filename='pre_output.log', filemode='w',
                    level=logging.INFO)


'''
flow:   main (skip isotropic resampling)
        > create folders
        > calculate_weight
            > compute_distance_weight_matrix
                * from Lessman
                - distance_to_border = ndimage.distance_transform_edt(mask > 0)
            - write weight image
        > crop_unref_vert (remove vertebrae not labeled in ground truth)
        * in this dataset, all are labeled except for sacrum and coccyx
        * because we have no ground truth of completeness for vertebra,
        * just crop all first/last, label those as incomplete, all else complete
            > findZRange(img, mask)
                - get min/max vertebra
                > z_mid(mask, vert_low), z_mid(mask, vert_up)
                    - indices = np.nonzero(mask == chosen_vert (low/high))
                    lower = [np.min(i) for i in indices]
                    upper = [np.max(i) for i in indices]
                - z_range equals above
            - write cropped images
'''


# function for middle height value of a given vertebrae, should be x or shape[0]
def z_mid(mask, chosen_vert):
    mask_vert = mask[chosen_vert,:,:]
    indices = np.nonzero(mask_vert)  # coordinates for all points w/ nonzeros: [0] xdim (height), [1] ydim (width)
    lower = [np.min(i) for i in indices]  # min: [0] is height (x), [1] is width (y)
    upper = [np.max(i) for i in indices]  # max

    return int((lower[0] + upper[0]) / 2)  # only care about x dimension (height)


# find height range of all vertebra, dim shape[0]
def findZRange(mask, verts):
    # list available vertebrae
    vert_low = verts[1]  # be careful to avoid 0 (bg), this should be min vert
    vert_up = verts[-1]

    z_range = [z_mid(mask, vert_low), z_mid(mask, vert_up)]
    logging.info('Range of Z axis is: %s' % z_range)
    return z_range


def crop_unref_vert(path, out_path, subset, case):
    input_path = os.path.join(path, subset, case)
    output_path = os.path.join(out_path, subset, case)

    casepath = input_path
    casefiles = os.listdir(casepath)
    verts = []

    # first get mask - use this for finding z range
    mask = np.zeros((26, 1280, 640))  # max vert is 25
    for casefile in casefiles:
        if not casefile.startswith('vertebra'):
            continue  # skip image file
        # print('casefile is:', casefile)  # debug
        vertnum = int(casefile.split('_')[1])  # vertebra num
        # case 155 is messed up. vert 7 (lowest vert) is empty...
        if (subset == 'train' and case == '155' and vertnum == 7):
            print('training case 155 vertnum 7 will be skipped')
            logging.info('training case 155 vertnum 7 will be skipped')

            continue  # skip vert 7 for case 155 cause it shouldn't exist
        if (subset == 'train' and case == '228' and vertnum == 7):
            print('training case 228 vertnum 7 will be skipped')
            logging.info('training case 288 vertnum 7 will be skipped')
            continue
        verts.append(vertnum)
        itermask = np.array(Image.open(os.path.join(casepath, casefile)))
        mask[vertnum, :, :] = itermask  # now we have full 26-layer mask

    verts.sort()

    z_range = findZRange(mask, verts)

    for casefile in casefiles:
        casefilepath = os.path.join(casepath, casefile)
        outfilepath = os.path.join(output_path, casefile)

        raw = np.array(Image.open(casefilepath))
        cropped = raw[z_range[0]:z_range[1],:]
        img_pil = Image.fromarray(cropped)
        dst = outfilepath
        logging.info('saving file: %s' % dst)
        img_pil.save(dst)  # now we have cropped all files along new xdim (height) and saved


# calculate the weight via distance transform
def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=6):
    """
    Code from author : Dr.Lessman (nikolas.lessmann@radboudumc.nl)
    """
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
    weights = alpha + beta * np.exp(-(distance_to_border ** 2 / omega ** 2))
    return np.asarray(weights, dtype='float32')


def calculate_weight(data_uncropped_path, subset, case):
    # mask_path = os.path.join(data_uncropped_path, subset, case, 'seg')  # now casepath below
    # weight_path = os.path.join(data_uncropped_path, subset, case, 'weight')  # just save w prefix 'weight'

    # Path(mask_path).mkdir(parents=True, exist_ok=True)
    # Path(weight_path).mkdir(parents=True, exist_ok=True)

    # get mask from original data
    casepath = os.path.join(data_uncropped_path, subset, case)
    casefiles = os.listdir(casepath)
    # mask = np.zeros((26, 1280, 640))  # max vert is 25

    for casefile in casefiles:
        if not casefile.startswith('vertebra'):
            continue  # skip image file
        # print('casefile is:', casefile)  # debug
        vertnum = int(casefile.split('_')[1])  # vertebra num
        itermask = np.array(Image.open(os.path.join(casepath, casefile)))
        # mask[vertnum, :, :] = itermask  # now we have full 26-layer mask

        seg_mask = itermask
        weight = compute_distance_weight_matrix(seg_mask)
        image_pil = Image.fromarray(weight)
        dstfile = 'weight_' + casefile
        dst = os.path.join(casepath, dstfile)
        image_pil.save(dst)

        logging.info('Calculating weight of %s' % dst)


def create_folders(root, subset, case, folders):
    filepath = os.path.join(root, subset, case)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    logging.info('Making directory %s' % filepath)


def main():
    ddir = './data'
    output_crop = ddir
    folders = ['img', 'seg', 'weight']
    subsets = ['train', 'val', 'test']

    counter = 0  # 603 total files
    for subset in subsets:
        # get existing cases from uncropped directory
        cases = os.listdir(os.path.join('./data_uncropped', subset))
        cases.sort()

        for case in cases:
            counter += 1
            print('Processing case', counter, 'of 603')  # total 603 cases
            print('Subset is',subset)
            print('Casenum is:', case)
            logging.info('Processing case %s of 603' % counter)
            logging.info('Subset is %s' % subset)
            logging.info('Casenum is: %s' % case)

            # uncomment line below if need to create folders again arises
            create_folders(output_crop, subset, case, folders)

            data_uncropped_path = './data_uncropped'

            # uncomment line below to calculate the weight
            # calculate_weight(data_uncropped_path, subset, case)

            output_path = ddir
            # uncomment line below to crop start and end vertebra of each spine
            crop_unref_vert(data_uncropped_path, output_path, subset, case)


if __name__ == '__main__':
    main()