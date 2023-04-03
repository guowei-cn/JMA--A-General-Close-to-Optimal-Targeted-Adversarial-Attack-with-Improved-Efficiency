# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('gtsrb/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

# import matplotlib.pyplot as plt
import csv
import numpy as np
import os

DATASET_ROOT = r'G:\bowen\Datasets'


# function for reading the images
# arguments: path to the traffic sign data, for example './gtsrb/Training'
# returns: list of images, list of corresponding labels
def readTrainingTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './gtsrb/Training'
    Returns:   list of images, list of corresponding labels'''
    # images = [] # images
    img_paths =[] # img paths
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.__next__() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            # images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            img_paths.append(prefix + row[0])
            labels.append(row[7]) # the 8th column is the label # gtsrb label
        gtFile.close()
    return img_paths, labels

def readTestingTrafficSigns(rootpath):
    img_paths = []
    labels = []

    gtFile = open(rootpath + '/' + 'GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.__next__() # skip header
    for row in gtReader:
        img_paths.append(rootpath + '/' + row[0])
        labels.append(row[7])  # the 8th column is the label # gtsrb label
    gtFile.close()
    return img_paths, labels


def BCH_labels(root_path, bit_model_index):
    # bit_model_index actually starts from 1
    # original gtsrb file organize, imgs with csv in same folder
    if 'Training' in root_path:
        img_path_list, gtsrb_label_list = readTrainingTrafficSigns(root_path)
    elif 'Test' in root_path:
        img_path_list, gtsrb_label_list = readTestingTrafficSigns(root_path)
    else:
        raise TypeError('Check data folder that has Test or Training in its name')

    # load label_code_matrix
    xxx = np.load(os.path.join(DATASET_ROOT, r'gtsrb\n15-k5-t3-BCH-select_label-code_matrix.npy')) # path to that matrix whose first column is gtsrb label, other 32 rows are bit model label
    img_paths = np.array(img_path_list)
    gtsrb_labels = np.array(gtsrb_label_list, dtype=np.int)

    vaild_indexes =  np.array([i in xxx[:,0] for i in gtsrb_labels])
    valid_img_paths = img_paths[vaild_indexes]
    valid_gtsrb_labels = gtsrb_labels[vaild_indexes]

    bit_model_labels = valid_gtsrb_labels.copy()
    if bit_model_index == 0:
        # for 0-31 label that used for integrate bch_latent model and surrogate model
        for idx, i in enumerate(xxx[:, 0]):
            # i --> idx
            bit_model_labels = np.where(valid_gtsrb_labels == i, idx, bit_model_labels)
    else:
        for idx, i in enumerate(xxx[:, 0]):
            # bit_model_indexes starts from the second column i.e. index 1 for the first row is ori gtsrb label
            bit_model_labels = np.where(valid_gtsrb_labels == i, xxx[idx, bit_model_index], bit_model_labels)

    return valid_img_paths, bit_model_labels

def Hadamard_labels(image_root_folder, valid_gtsrb_categories, hadamard_matrix, bit_index):
    """

    :param image_root_folder: either train folder or test folder provided by gtsrb as it is
    :param valid_gtsrb_categories: e.g. 32 categories of gtsrb who has largest num of img examples
    :param hadamard_matrix: ndarray of codeword matrix
    :param bit_index: index of bit model, starts from 1, also the index of bit in one codeword
    :return: img_path_list for valid_gtsrb_categories, corresponds hadamard labels for number bit_index bit/model
    """
    # original gtsrb file organize, imgs with csv in same folder
    if 'Training' in image_root_folder:
        img_path_list, gtsrb_label_list = readTrainingTrafficSigns(image_root_folder)
    elif 'Test' in image_root_folder:
        img_path_list, gtsrb_label_list = readTestingTrafficSigns(image_root_folder)
    else:
        raise TypeError('Check data folder that has Test or Training in its name')
    img_paths = np.array(img_path_list)
    gtsrb_labels = np.array(gtsrb_label_list, dtype=np.int)

    # only gtsrb categories that appears in valid categories will be record/picked
    vaild_indexes =  np.array([i in valid_gtsrb_categories for i in gtsrb_labels])
    valid_img_paths = img_paths[vaild_indexes]
    valid_gtsrb_labels = gtsrb_labels[vaild_indexes]

    # prepare labels according to bit_index
    bit_model_labels = np.empty(valid_gtsrb_labels.shape)
    if bit_index <= 0:
        # return numerical labels that 0-31, used for training normal multi-class classification (surrogate)
        for idx, gtsrb_label in enumerate(valid_gtsrb_categories):
            # gtsrb_label -> numerical labels
            bit_model_labels = np.where(valid_gtsrb_labels == gtsrb_label, idx, bit_model_labels)
    else:
        for idx, gtsrb_label in enumerate(valid_gtsrb_categories):
            bit_model_labels = np.where(valid_gtsrb_labels == gtsrb_label, hadamard_matrix[idx, bit_index-1], bit_model_labels)

    return valid_img_paths, bit_model_labels


if __name__ == '__main__':

    MODEL_ROOT = r'G:\bowen'
    DATASET_ROOT = r'G:\bowen\Datasets'

    TOP32GTSRB_CATEGORIES = np.load(os.path.join(DATASET_ROOT, r'gtsrb/gysrb_top32category_label.npy'))
    TEST_IMG_FOLDER = os.path.join(DATASET_ROOT, r'gtsrb/Final_Test/Images')

    HADAMARD_MATRIX = np.load('hadamard32.npy')

    test_img_paths, bit_labels = Hadamard_labels(TEST_IMG_FOLDER,
                                                 TOP32GTSRB_CATEGORIES,
                                                  HADAMARD_MATRIX, 15)
    print(test_img_paths, bit_labels)