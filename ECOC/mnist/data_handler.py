"""
mapping between mnist labels and hadamard labels, as well as to be attacked images
"""
import os
from glob import glob
import numpy as np
from ECOC.mnist.models import ECOC_Hadamard_Model
from keras.datasets import mnist

MODEL_DIR = '/media/benedetta/Barracuda 2TB/bzhang/trained_models/'

def mnist_to_hadamard(hadamard_matrix):
    # .load_data() gives image array in [0,255]
    (x_train, y_train_mnist), (x_test, y_test_mnist) = mnist.load_data()
    x_train_val = np.pad(x_train, [(0,0),(2,2),(2,2)], mode='constant')
    x_test = np.pad(x_test, [(0,0),(2,2),(2,2)], mode='constant')
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    # x_test.shape = (10000, 32, 32,3) y_test.shape = (10000,1)
    y_train_hadamard = np.squeeze(hadamard_matrix[y_train_mnist, :])
    y_test_hadamard = np.squeeze(hadamard_matrix[y_test_mnist, :])
    return x_train/255, y_train_hadamard, x_test/255, y_test_hadamard

def get_tobe_attacked_imgs(hadamard_matrix, mnist_testset_index=None):

    (x_train, y_train_mnist), (x_test, y_test_mnist) = mnist.load_data()

    x_test = np.pad(x_test, [(0,0),(2,2),(2,2)], mode='constant')
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    if mnist_testset_index is not None:
        imgs = x_test[mnist_testset_index]
        mnist_labels = np.squeeze(y_test_mnist[mnist_testset_index])
    else:
        imgs = x_test
        mnist_labels = np.squeeze(y_test_mnist)

    hadamard_codewords = np.squeeze(hadamard_matrix[mnist_labels, :])
    imgs = imgs / 255

    # print('-------- Loading to-be attacked images result-------\n'
    #       'imgs shape: {}\n'
    #       'mnist_labels shape: {}\n'
    #       'hadamard_codewords shape: {}\n'
    #       .format(imgs.shape, mnist_labels.shape, hadamard_codewords.shape))

    return imgs, mnist_labels, hadamard_codewords


def get_error_pattern(ecoc_model):
    imgs, mnist_labels, hadamard_codewords = get_tobe_attacked_imgs()
    error_patterns = np.zeros_like(hadamard_codewords)
    target_labels = []
    for idx, (ground_label, ground_code) in enumerate(zip(mnist_labels, hadamard_codewords)):
        error = np.ones(16)
        decoded = error * ground_code
        while np.equal(ecoc_model._codeword2label(decoded), ground_label):
            error = np.ones(16)
            error[1 + np.random.choice(15, 5, replace=False)] = -1
            target_code = error * ground_code
            decoded = ecoc_model._decoding(target_code)
        error_patterns[idx] = error
        target_labels.append(ecoc_model._codeword2label(decoded))
    return error_patterns, np.array(target_labels)


def find_imgs_according_to_targets(hadamard_matrix, targeting_codeword, num = 50):
    target_label = np.where(np.all(hadamard_matrix == targeting_codeword, axis=1))[0]
    (x_train, y_train_mnist), (x_test, y_test_mnist) = mnist.load_data()

    x_test = np.pad(x_test, [(0,0),(2,2),(2,2)], mode='constant')
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    indexs = np.where(y_test_mnist==target_label)[0]
    selected_indexs = np.random.choice(indexs, num, replace=False)
    images = x_test[selected_indexs]

    return images/255


if __name__ == '__main__':

    bit_model_weights_paths = glob(os.path.join(MODEL_DIR,
        'mnist/ECOC/Hadamard16_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5'))
    bit_model_weights_paths.sort(key=lambda x: int(x.split('bit_')[-1].replace("\\", "/").split('/')[0]))
    print(bit_model_weights_paths)
    HADAMARD_MATRIX = np.load('david_code/Jacobian/hadamard16.npy')
    ecoc_model = ECOC_Hadamard_Model(bit_model_weights_paths, HADAMARD_MATRIX[:10, :])

    patterns, target_classes, = get_error_pattern(ecoc_model)
    print('patterns.shape', patterns.shape)
    print(np.sum(patterns, axis=1))
    np.save(os.path.join(MODEL_DIR, 'mnist/ECOC/5bits_valid_error_patterns_forHadamard16_surrogate_weights_freeze6.npy'), patterns)

