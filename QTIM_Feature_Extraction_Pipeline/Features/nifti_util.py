from __future__ import division
import GLCM
import numpy as np
import nibabel as nib
import os
from shutil import copy, move
import matplotlib.pyplot as plt
import glob

def copy_files(infolder, outfolder, name, duplicate=True):

    """ I'm not sure how many of these file-moving helper functions should be
        included. I know I like to have them, but it may be better to have 
        people code their own. It's hard to customize such functions exactly
        to users' needs.
    """

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

def return_nifti_attributes(filepath):

    """ For now, this just returns pixel dimensions, which are important
        for calculating volume etc. It is still unknown whether pixel
        dimensions are transformed by an affine matrix; they may need to
        be transformed back in the case of our arrays. TO-DO: Test.
    """

    img_nifti = nib.load(filepath)
    pixdim = img_nifti.header['pixdim']
    return pixdim[1:4]

def nifti_2_numpy(filepath):

    """ There are a lot of repetitive conversions in the current iteration
        of this program. Another option would be to always pass the nibabel
        numpy class, which contains image and attributes. But not everyone
        knows how to use that class, so it may be more difficult to troubleshoot.
    """

    img = nib.load(filepath).get_data()
    return img

def dcm_2_numpy(filepath):

    # Maybe for the future...

    return

def match_array_orientation(image1, image2):

    """ Flipping images and labels is often necessary, but also often
        idiosyncratic to the two images being flipped. It's an open question
        whether the flipping can be automatically determined. If it can, this
        function will do it; if not, other parameters will have to be added by the user.
        REMINDER: This will also have to change an image's origin, if that
        image has any hope of being padded correctly in a subsequent step.
    """

    image1_nifti = nib.load(image1)
    image2_nifti = nib.load(image2)

    image1_array = image1_nifti.get_data()
    image2_array = image2_nifti.get_data()

    image1_orientation = [image1_nifti.affine[0,0], image1_nifti.affine[1,1], image1_nifti.affine[2,2]]
    image2_orientation = [image2_nifti.affine[0,0], image2_nifti.affine[1,1], image2_nifti.affine[2,2]]

    return

def pad_nifti_image(image):

    """ Many people store their label maps in Niftis with dimensions smaller
        than the corresponding image. This is also that natural output of DICOM-SEG
        nifti conversions. Padding these arrays with empty values so that they are
        comparable requires knowledge of that image's origin.
    """

    return

def mask_nifti(image_numpy, label_numpy, label_indices):

    masked_images = []

    for idx in label_indices[1:]:
        masked_image = np.copy(image_numpy)
        masked_image[label_numpy != idx] = 0
        masked_image = truncate(masked_image)
        masked_images += [masked_image]

    return masked_images

def truncate(image_numpy):

    """ Filed To: This Is So Stupid
        There are better ways online to do what I am attempting,
        but so far I have not gotten any of them to work. In the meantime,
        this long and probably ineffecient code will suffice. It is
        meant to remove empty rows from images.
    """

    dims = image_numpy.shape
    truncate_range_x = [0,dims[0]]
    truncate_range_y = [0,dims[1]]
    truncate_range_z = [0,dims[2]]
    start_flag = True

    for x in xrange(dims[0]):
        if np.sum(image_numpy[x,:,:]) == 0:
            if start_flag:
                truncate_range_x[0] = x + 1
        else:
            start_flag = False
            truncate_range_x[1] = x + 1

    start_flag = True

    for y in xrange(dims[1]):
        if np.sum(image_numpy[:,y,:]) == 0:
            if start_flag:
                truncate_range_y[0] = y + 1
        else:
            start_flag = False
            truncate_range_y[1] = y + 1

    start_flag = True

    for z in xrange(dims[2]):
        if np.sum(image_numpy[:,:,z]) == 0:
            if start_flag:
                truncate_range_z[0] = z + 1
        else:
            start_flag = False
            truncate_range_z[1] = z + 1

    truncate_image_numpy = image_numpy[truncate_range_x[0]:truncate_range_x[1], truncate_range_y[0]:truncate_range_y[1], truncate_range_z[0]:truncate_range_z[1]]


    return truncate_image_numpy

def coerce_levels(image_numpy, levels=255, method="divide"):

    """ In volumes with huge outliers, the divide method will
        likely result in many zero values. This happens in practice
        quite often. TO-DO: find a better method to bin image values.
        I'm sure there are a thousand such algorithms out there to do
        so. Maybe something based on median's, rather than means. This,
        of course, loses the 'Extremeness' of extreme values. An open
        question of how to reconcile this -- maybe best left to the user.
    """

    levels -= 1
    if method == "divide":
        image_max = np.max(image_numpy)
        for i in xrange(image_numpy.shape[2]):
            image_slice = image_numpy[:,:,i]
            image_numpy[:,:,i] = np.round((image_slice / image_max) * levels)
        return image_numpy

def coerce_positive(image_numpy):

    """ Required by GLCM. Not sure of the implications for other algorithms.
    """

    image_min = np.min(image_numpy)
    if image_min < 0:
        image_numpy = image_numpy + image_min
    return image_numpy

def check_image(image_numpy, second_image_numpy=[], mode="cycle", step=1):

    """ A useful utiltiy for spot checks.
    """

    if second_image_numpy != []:
        for i in xrange(image_numpy.shape[0]):
            print i
            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(image_numpy[:,:,i*step], interpolation='none', aspect='auto')
            a=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(second_image_numpy[:,:,i*step], interpolation='none', aspect='auto')
            plt.show()
    else:
        if mode == "cycle":
            for i in xrange(image_numpy.shape[0]):
                fig = plt.figure()
                imgplot = plt.imshow(image_numpy[i,:,:], interpolation='none', aspect='auto')
                plt.show()

        if mode == "first":
            fig = plt.figure()
            imgplot = plt.imshow(image_numpy[0,:,:], interpolation='none', aspect='auto')
            plt.show()


if __name__ == '__main__':
    grab_files('C:/Users/azb22/Documents/Scripting/Head_Neck_Cancer_Challenge/Training/Training/Training/Case_10', 'C:/Users/azb22/Documents/Scripting/Head_Neck_Cancer_Challenge/Training/Training/Training/Case_10/TempFolder', '*dcm')