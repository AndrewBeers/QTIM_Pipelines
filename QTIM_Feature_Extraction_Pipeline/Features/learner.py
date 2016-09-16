import GLCM
import nifti_util
import glob
import os
import numpy as np
import csv

feature_dictionary = {'GLCM': GLCM}

def features_batch(folder, features=['GLCM'], labels=False, label_suffix="-label", decisions=False, levels=255, filenames=True, featurenames=True, outfile=''):
    
    imagepaths = glob.glob(os.path.join(folder, "*.nii*"))
    imagepaths = [ x for x in imagepaths if label_suffix not in x ]
    if labels:
        label_images = glob.glob(os.path.join(folder, "*" + label_suffix + ".nii*"))

    if imagepaths == []:
        raise ValueError("There are no .nii or .nii.gz images in the provided folder.")
    if labels and label_images == []:
        raise ValueError("There are no labels with the provided suffix in this folder. If you do not want to use labels, set the \'labels\' flag to \'False\'. If you want to change the label file suffix (default: \'-label\'), then change the \'label_suffix\' flag.")

    total_features = 0
    feature_indexes = [0]

    for feature in features:
        total_features += feature_dictionary[feature].feature_count()
        if feature_indexes == [0]:
            feature_indexes = [0, feature_dictionary[feature].feature_count()]
        else:
            feature_indexes = feature_indexes[-1] + [feature_dictionary[feature].feature_count()]
    
    image_list = []
    imagename_list = []

    for imagepath_idx, imagepath in enumerate(imagepaths):
        if labels:
            split_path = str.split(imagepath, '.')
            label_path = split_path[0] + label_suffix + '.' + '.'.join(split_path[1:])
            print label_path

            if os.path.isfile(label_path):
                image = nifti_util.nifti_2_numpy(imagepath)
                label_image = nifti_util.nifti_2_numpy(label_path)
                label_indices = np.unique(label_image)

                if label_indices.size == 1:
                    print 'Warning: image at path ' + imagepath + ' has an empty label-map, and will be skipped.'
                    continue

                masked_images = nifti_util.mask_nifti(image, label_image, label_indices)
                image_list += masked_images

                filename = str.split(label_path, '\\')[-1]

                if label_indices.size == 2:
                    imagename_list += [filename]
                else:
                    split_filename = str.split(filename, '.')
                    for labelval in label_indices[1:]:
                        filename = split_filename[0] + '_' + str(labelval) + '.' + split_filename[1]
                        imagename_list += [filename]

            else:
                print 'Warning: image at path ' + imagepath + ' has no label-map, and will be skipped.'
                continue

        else:
            image = nifti_util.nifti_2_numpy(imagepath)
            image_list += [image]
            imagename_list += [imagepath]

    if image_list == []:
        raise ValueError("Images and labels are mismatched, or all labels are empty. No features will be extracted.")
    
    numerical_output = np.zeros((len(image_list), total_features), dtype=float)

    for image_idx, image in enumerate(image_list):

        print np.sum(image)
        print image.shape

    for image_idx, image in enumerate(image_list):


        if levels > 0:
            image = nifti_util.coerce_levels(image, levels=levels)

        # image = np.random.randint(0,6,(10,10,10))

        for feature_idx, feature in enumerate(features):

            if feature == 'GLCM':
                if levels <= 0 or levels > 255:
                    nifti_util.coerce_levels(image, levels=255)
                if np.min(image) < 0:
                    nifti_util.coerce_positive(image)
                # numerical_output[image_idx, feature_indexes[feature_idx]:feature_indexes[feature_idx+1]] = GLCM.glcm_features(image, levels=6)
                numerical_output[image_idx, feature_indexes[feature_idx]:feature_indexes[feature_idx+1]] = GLCM.glcm_features(image, levels=levels)

    index_output = np.zeros((len(image_list),1), dtype="object")
    if filenames:
        index_output[:,0] = imagename_list
    else:
        index_output[:,0] = range(len(image_list))

    final_output = np.hstack((index_output, numerical_output))

    if featurenames:
        label_output = np.zeros((1, total_features+1), dtype=object)
        for feature_idx, feature in enumerate(features):
            label_output[0, (1+feature_indexes[feature_idx]):(1+feature_indexes[feature_idx+1])] = feature_dictionary[feature].featurename_strings()
        label_output[0,0] = ''

    print final_output
    if outfile != '':
        with open(outfile, 'wb') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            csvfile.writerow(label_output[0,:])
            for row in final_output:
                csvfile.writerow(row)
        # np.savetxt(outfile, final_output, delimiter=',', header=np.ravel(label_output))

    return features_batch

if __name__ == "__main__":
    features_batch(folder='C:/Users/azb22/Documents/GitHub/QTIM_Pipelines/QTIM_Feature_Extraction_Pipeline/Test_Data', labels=True, levels=10, outfile='C:/Users/azb22/Documents/GitHub/QTIM_Pipelines/QTIM_Feature_Extraction_Pipeline/Test_Data/Results.csv')