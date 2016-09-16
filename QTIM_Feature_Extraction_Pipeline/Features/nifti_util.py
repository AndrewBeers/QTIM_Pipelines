from __future__ import division
import GLCM
import numpy as np
import nibabel as nib
import os
from shutil import copy, move
import glob

def copy_files(infolder, outfolder, name, duplicate=True):
    path = os.path.join(infolder, name)
    print path
    files = glob.glob(path)
    if files == []:
        print 'No files moved. Might you have made an error with the filenames?'
    else:
        for file in files:
            if duplicate:
                copy(file, outfolder)
            else:
                move(file, outfolder)

def nifti_2_numpy(filepath):
    img = nib.load(filepath).get_data()
    return img

def match_array_orientation(image1, image2):
    image1_nifti = nib.load(image1)
    image2_nifti = nib.load(image2)

    image1_array = image1_nifti.get_data()
    image2_array = image2_nifti.get_data()

    image1_orientation = [image1_nifti.affine[0,0], image1_nifti.affine[1,1], image1_nifti.affine[2,2]]
    image2_orientation = [image2_nifti.affine[0,0], image2_nifti.affine[1,1], image2_nifti.affine[2,2]]

    return

def pad_nifti_image(image):

    return

def mask_nifti(image_numpy, label_numpy, label_indices):

    masked_images = []

    for idx in label_indices[1:]:
        print np.sum(image_numpy)
        print idx
        masked_image = np.copy(image_numpy)
        masked_image[label_numpy != idx] = 0
        masked_image = truncate(masked_image)
        masked_images += [masked_image]
        print np.sum(masked_image)

    return masked_images

def truncate(image_numpy):

    # mask = image_numpy == 0
    # print mask.shape
    # all_white = mask.sum(axis=2) == 0
    # print all_white.shape
    # rows = np.flatnonzero((~all_white).sum(axis=1))
    # cols = np.flatnonzero((~all_white).sum(axis=0))

    # image_numpy = image_numpy[:, np.all(image_numpy != 0, axis=0)]
    # print image_numpy
    # print image_numpy.shape

    # crop = image_numpy[rows.min():rows.max()+1, cols.min():cols.max()+1, :]
    # print crop.shape
    return image_numpy

def coerce_levels(image_numpy, levels=255, method="divide"):
    levels -= 1
    if method == "divide":
        image_max = np.max(image_numpy)
        for i in xrange(image_numpy.shape[2]):
            image_slice = image_numpy[:,:,i]
            image_numpy[:,:,i] = np.round((image_slice / image_max) * levels)
        return image_numpy

def coerce_positive(image_numpy):
    image_min = np.min(image_numpy)
    if image_min < 0:
        image_numpy = image_numpy + image_min
    return image_numpy





if __name__ == '__main__':
    grab_files('C:/Users/azb22/Documents/Scripting/Head_Neck_Cancer_Challenge/Training/Training/Training/Case_10', 'C:/Users/azb22/Documents/Scripting/Head_Neck_Cancer_Challenge/Training/Training/Training/Case_10/TempFolder', '*dcm')