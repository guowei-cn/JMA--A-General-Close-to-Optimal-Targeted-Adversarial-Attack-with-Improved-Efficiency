"""
mapping between cifar10 labels and hadamard labels, as well as to be attacked images
"""
import sys
import os
import numpy as np
from keras.datasets import cifar10

MODEL_ROOT = r'G:\bowen'
DATASET_ROOT = r'G:\bowen\Datasets'

def cifar_to_hadamard(hadamard_matrix):
    # .load_data() gives image array in [0,255]
    (x_train, y_train_cifar), (x_test, y_test_cifar) = cifar10.load_data()
    # x_test.shape = (10000, 32, 32,3) y_test.shape = (10000,1)
    y_train_hadamard = np.squeeze(hadamard_matrix[y_train_cifar, :])
    y_test_hadamard = np.squeeze(hadamard_matrix[y_test_cifar, :])
    return x_train/255, y_train_hadamard, x_test/255, y_test_hadamard

def get_tobe_attacked_imgs(hadamard_matrix, cifar_testset_index):
    _, (x_test, y_test_cifar) = cifar10.load_data()
    imgs = x_test[cifar_testset_index]
    cifar_labels = np.squeeze(y_test_cifar[cifar_testset_index])
    hadamard_codewords = np.squeeze(hadamard_matrix[cifar_labels, :])
    imgs = imgs / 255

    # print('-------- Loading to-be attacked images result-------\n'
    #       'imgs shape: {}\n'
    #       'cifar_labels shape: {}\n'
    #       'hadamard_codewords shape: {}\n'
    #       .format(imgs.shape, cifar_labels.shape, hadamard_codewords.shape))

    return imgs, cifar_labels, hadamard_codewords

def find_imgs_according_to_targets(hadamard_matrix, targeting_codeword, num = 50):
    target_cifar_label = np.where(np.all(hadamard_matrix == targeting_codeword, axis=1))[0]
    (x_train_val, y_train_val_cifar), (x_test, y_test_cifar) = cifar10.load_data()

    indexs = np.where(y_test_cifar==target_cifar_label)[0]
    selected_indexs = np.random.choice(indexs, num, replace=False)
    images = x_test[selected_indexs]

    return images/255