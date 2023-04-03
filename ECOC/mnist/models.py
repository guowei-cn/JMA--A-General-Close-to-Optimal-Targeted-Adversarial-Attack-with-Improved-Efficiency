""" mnist networks structures that used
use keras 2.3.1 with tf2 backend, not tf directly
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input, Concatenate, Conv2D, Activation, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
import os


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

if 'media' in os.path.dirname(os.path.abspath(__file__)):
    MODEL_DIR = '/media/hdddati2/bzhang/trained_models'
else:
    MODEL_DIR = r'G:\bowen'

def get_1bit_ensemble_vgg16(model_weights_path=None, input_tensor=None):
    bottom = VGG16(include_top=False, weights=None, input_shape=(32, 32, 1), input_tensor=input_tensor)
    flatten = Flatten()(bottom.output)
    fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
    output = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(fc2)
    model = Model(inputs=bottom.input, outputs=output)
    if model_weights_path:
        model.load_weights(model_weights_path)
    return model

def get_10class_surrogate_vgg16(model_weights_path=None, input_tensor=None):
    bottom = VGG16(include_top=False, weights=None, input_shape=(32, 32, 1), input_tensor=input_tensor)
    flatten = Flatten()(bottom.output)
    fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
    output = Dense(units=10, kernel_initializer="he_normal", activation='softmax', name='output')(fc2)
    model = Model(inputs=bottom.input, outputs=output)
    if model_weights_path:
        model.load_weights(model_weights_path)
    return model

class ECOC_Hadamard_Model():

    def __init__(self, bit_model_weight_path_in_order, hadamard_matrix):
        # print('models are prepared in order, i.e.')
        # for idx, path in enumerate(bit_model_weight_path_in_order):
        #     print('Nuber {} model/bit is {}'.format(idx, path))
        self.__keras_model = self.build_ge_model(bit_model_weight_path_in_order)
        self.__hadamard_matrix = hadamard_matrix

    # properties
    @property
    def keras_model(self):
        return self.__keras_model
    @property
    def hadamard_matrix(self):
        return self.__hadamard_matrix
    @property
    def num_class(self):
        return self.__hadamard_matrix.shape[0]

        # build model
    def build_chuan_model(self, bit_model_weight_path_in_order):
        # Chinese character 川 that all models works independently but sharing one same input image
        # base models been stored into a list, needs test
        input = Input((32, 32, 1), name='very_input')

        bit_models = []
        for idx, path in enumerate(bit_model_weight_path_in_order):
            base = get_1bit_ensemble_vgg16(path, input_tensor=input)
            model = Model(inputs=input, outputs=base.outputs)
            for layer in model.layers:
                layer.name = 'bit{}_'.format(idx+1) + layer.name
            bit_models.append(model)

        cat_model = Model(input = input,
                          outputs = Concatenate()(
                              [model.layers[-1].output for model in bit_models]
                          ))

        return cat_model

    def build_ge_model(self, bit_model_weight_path_in_order):
        # Chinese character 个 that all models share some bottom layer and input image, while work independently later at topper layers
        def get_ge_shared_and_unique_parts( num_bottom_layers, output_model_type, full_model_weight_path=None, input_tensor=None):
            """

            :param full_model_weight_path:
            :param input_tensor: must be passed if shared bottom is desired
            :param num_bottom_layers:  layers that should be abandon to get branch
            :return:
            """
            bottom = VGG16(include_top=False, weights=None, input_shape=(32, 32, 1), input_tensor=input_tensor)
            flatten = Flatten()(bottom.output)
            fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
            fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
            output = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(fc2)
            model = Model(inputs=bottom.input, outputs=output)

            if full_model_weight_path:
                model.load_weights(full_model_weight_path)
            else:
                model.load_weights('Model/mnist/model_weight/Hadamard16_surrogate_weights_freeze6_bit_1/final_trained_weights.hdf5')

            if output_model_type == 'bottom':
                shared_bottom = Model(inputs = model.inputs,
                               outputs = model.layers[num_bottom_layers].output) # todo considering add name to models
                # unique_branch = Model(inputs = model.layers[num_bottom_layers+1].input,
                #                       outputs = model.output)
                return shared_bottom
            elif output_model_type == 'branch':
                branch_input = Input(model.layers[num_bottom_layers+1].input_shape[1:])
                unique_branch = branch_input
                for layer in model.layers[num_bottom_layers+1:]:
                    unique_branch = layer(unique_branch)
                unique_branch = Model(inputs=branch_input, outputs=unique_branch)
                return unique_branch
            else:
                raise TypeError('output_model_type has to be either bottom or branch')

        num_freezed_layer = int(bit_model_weight_path_in_order[0].split('freeze')[1].split('_')[0])
        very_input = Input((32, 32, 1), name='very_input')
        # the bottom x layers that shared by all ensemble branches, if no weights specified then 32-class surrogate model weights is by default used
        shared_bottom= get_ge_shared_and_unique_parts(num_freezed_layer, output_model_type='bottom', input_tensor = very_input)

        bit_top_branches = []
        for idx, path in enumerate(bit_model_weight_path_in_order):
            # top y layers that is unique for every branches, well takes the same input, always need pass new weights
            branch = get_ge_shared_and_unique_parts(num_freezed_layer, output_model_type='branch', full_model_weight_path=path, input_tensor=very_input)
            for layer in branch.layers:
                layer._name = 'bit{}_'.format(idx+1) + layer._name
            bit_top_branches.append(branch)

        # reconnecting everything
        reconnect_branch_ends = []
        for bit_branch in bit_top_branches:
            connected = shared_bottom.output
            for idx, layer in enumerate(bit_branch.layers):
                if idx==0:
                    continue
                connected = layer(connected)
            reconnect_branch_ends.append(connected)

        cat_model = Model(inputs=shared_bottom.inputs,
                          outputs=Concatenate()(
                              [bit_branch_end for bit_branch_end in reconnect_branch_ends]
                          ))
        # end reconnection

        # build-in functions construction
        # logits function
        individual_activations = cat_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self._logits_func = K.function([cat_model.input], logits)
        # build-in losses function
        y_trues = K.placeholder(shape=cat_model.output_shape[1])
        hinge_losses = [tf.keras.losses.hinge(y_pred=y_pred, y_true=y_trues[i]) for i, y_pred in enumerate(individual_activations)]
        self._hinge_losses_func = K.function([cat_model.input, y_trues], hinge_losses)

        return cat_model

    def _decoding(self, raw_codewords, decode_method ='min_distance'):
        """
        map un decoded codeword to legal codeword
        :param raw_codewords:
        :param decode_method:
        :return:
        """
        # for codewords in {-1, 1}, min distance decoding and correlation decoding is the same
        dp_raw_codeword = np.matmul(raw_codewords, self.hadamard_matrix.T)
        index_labels = np.argmax(dp_raw_codeword, axis=dp_raw_codeword.ndim - 1)
        md_codeword = self.hadamard_matrix[index_labels]
        return md_codeword

    def _codeword2label(self, codewords):
        """
        map decoded codeword to hard label
        :param codewords:
        :return:
        """
        dp_raw_codeword = np.matmul(codewords, self.hadamard_matrix.T)
        index_labels = np.argmax(dp_raw_codeword, axis= -1)
        # label = np.argmax((ecoc_model.hadamard_matrix == codewords).all(1))
        # label_1 = np.argmax(np.matmul(codewords, self.hadamard_matrix.T), axis=-1)
        # assert label == label_1
        return index_labels


    def predict(self, img, output_type = 'activated'):
        # todo require more tests
        """
        predict image as a concatnate model, so the output size should be 32 since it't tanh activation and hinge loss
        :param img:
        :return:
        """
        # outputs that been through activation (tanh, -1,1)
        if img.ndim == 3:
            img = img[np.newaxis]
        outputs = self.keras_model.predict(img)  # shape should be (,32)
        if output_type in ['logits', 'logit', 'unactivated']:
            logits = np.array(self._logits_func([img])).squeeze()
            return logits
        if output_type == 'activated':
            return outputs
        undecoded = np.sign(outputs)
        if output_type == 'undecoded':
            return undecoded

        # ECOC stuff
        # calculate dot product with all hadamard codewords to get what I called dot product logits
        dp_logits = np.matmul(outputs, self.hadamard_matrix.T)  # shape: (num_img, 16) * (16, 10), hadamard matrix.T

        if output_type in ['dot_product_logits', 'ecoc_logits']:
            return dp_logits

        codeword = np.squeeze(self.hadamard_matrix[np.argmax(dp_logits, axis=dp_logits.ndim-1)])
        # todo correlation decoding is not the best, thus index label accuracy is not so good, add one minimum distance decoding
        if output_type in ['codewords' , 'codeword' ,'decoded', 'ecoc_codeword', 'ecoc_decoded']:
            return codeword
        # kind of softmax
        if output_type in ['probability', 'probabilities']:
            dp_logits_positive = np.where(dp_logits < 0, 0, dp_logits)
            probabilities = (dp_logits_positive.T / np.sum(dp_logits_positive, axis=dp_logits_positive.ndim - 1)).T
            return probabilities

        # MINIMUM DISTANCE decoding, only works for hadamard now
        if output_type in ['min_dis_decoded', 'min_dis']:
            dp_raw_codeword = np.matmul(undecoded, self.hadamard_matrix.T)
            index_labels = np.argmax(dp_raw_codeword, axis=dp_raw_codeword.ndim-1)
            md_codeword = np.squeeze(self.hadamard_matrix[index_labels])
            return md_codeword

        if output_type in ['min_dis_label', 'min_dis_class', 'min_dis_hard_label']:
            dp_raw_codeword = np.matmul(undecoded, self.hadamard_matrix.T)
            index_labels = np.argmax(dp_raw_codeword, axis=dp_raw_codeword.ndim-1)
            return index_labels

        if output_type in ['ecoc_label', 'ecoc_class', 'ecoc_hard_label', 'hard_label']:
            # dp_raw_codeword = np.matmul(undecoded, self.hadamard_matrix.T)
            # print('--------inside ecoc_model.predict output_type = ecoc_hard_label-----\n')
            index_labels = np.argmax(dp_logits, axis=dp_logits.ndim - 1)
            return index_labels




if __name__ == '__main__':
    # module test

    """ rebuild surrogate model and save weights with output replaced with bit node """
    # model = get_10class_surrogate_vgg16('/media/hdddati2/bzhang/trained_models/mnist/surrogate_vgg16/surrogate_vgg16_mnist_weights.h5')
    # model.summary()
    #
    #
    # print(model.get_layer('fc2').output)
    # print(model.input)
    # bottom = keras.models.Model(inputs=model.input, outputs = model.get_layer('fc2').output)
    # node_dense = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(bottom.output)
    # new_model = keras.models.Model(inputs = model.input, outputs = node_dense)
    # new_model.summary()
    #
    # print(new_model.output, new_model.output.op.inputs, new_model.output.op.inputs[0])
    #
    # new_model.save_weights('/media/hdddati2/bzhang/trained_models/mnist/surrogate_vgg16/surrogate_vgg16_mnist_final_layer_replaced_for_bit_model_weights.h5')
    #
    # bit_model = get_1bit_ensemble_vgg16('/media/hdddati2/bzhang/trained_models/mnist/surrogate_vgg16/surrogate_vgg16_mnist_final_layer_replaced_for_bit_model_weights.h5')
    # bit_model.summary()

    """ biuld ecoc model and test """
    from glob import glob
    bit_model_weights_paths = glob(os.path.join(MODEL_DIR,'mnist/ECOC/Hadamard16_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5'))
    bit_model_weights_paths.sort(key=lambda x: int(x.split('bit_')[-1].replace("\\","/").split('/')[0]))
    print(bit_model_weights_paths)
    HADAMARD_MATRIX = np.load('david_code/Jacobian/hadamard16.npy')
    ecoc_model = ECOC_Hadamard_Model(bit_model_weights_paths, HADAMARD_MATRIX[:10,:])
    print(ecoc_model.keras_model.summary(), ecoc_model.hadamard_matrix, ecoc_model.num_class)
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('before padding x_train.shape', x_train.shape)
    x_train = np.pad(x_train, [(0,0),(2,2),(2,2)], mode='constant')
    x_test = np.pad(x_test, [(0,0),(2,2),(2,2)], mode='constant')
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    # (x_train_val, y_train_val_cifar), (x_test, y_test_cifar) = cifar10.load_data()
    label_test_hadamard = np.squeeze(HADAMARD_MATRIX[y_test, :])
    imgs_test = x_test / 255
    idx = 0
    ecoc_success = []
    min_dis_success = []

    # batch prediction test
    activated = ecoc_model.predict(imgs_test, 'activated') # should be of shape (None, 16)
    codewords_1 = ecoc_model.predict(imgs_test, 'codewords')
    labels = np.argmax(np.matmul(activated, ecoc_model.hadamard_matrix.T), axis=1)
    codewords_2 = ecoc_model.hadamard_matrix[labels]
    assert np.array_equal(codewords_1, codewords_2)

    # undecoded = ecoc_model.predict(imgs_test, 'undecoded')
    # min_dis_decoded = np.squeeze(ecoc_model.predict(imgs_test, 'min_dis_decoded'))
    # min_dis_label = ecoc_model.predict(imgs_test, 'min_dis_label')

    ecoc_logits = ecoc_model.predict(imgs_test, 'ecoc_logits')
    ecoc_decoded = ecoc_model.predict(imgs_test, 'ecoc_decoded')
    ecoc_probability = ecoc_model.predict(imgs_test, 'probability')
    ecoc_label = ecoc_model.predict(imgs_test, 'ecoc_label')
    assert np.array_equal(labels, ecoc_label)

    # m_success_c = np.where(np.all(min_dis_decoded == label_test_hadamard,axis=1))[0]
    # m_success_l = np.where(min_dis_label == np.squeeze(y_test))[0]

    # e_success_c = np.where(np.all(ecoc_decoded == label_test_hadamard,axis=1))[0]
    # e_success_l = np.where(ecoc_label == np.squeeze(y_test))[0]
    # print('img set shape {}, label mnist shape {}, label hadamard shape {}'.format(imgs_test.shape, y_test.shape, label_test_hadamard.shape))
    # print('m_success_l', m_success_l)
    # print('np.array_equal(m_success_l, m_success_c)=',np.array_equal(m_success_l, m_success_c),
    #       '\nnp.array_equal(e_success_c,e_success_l)=', np.array_equal(e_success_c,e_success_l))
    #
    # print('MD ACC = {} / {} = {}'.format(len(m_success_l), len(y_test), len(m_success_l) / len(y_test)))
    # print('ECOC ACC = {} / {} = {}'.format(len(e_success_l), len(y_test), len(e_success_l) / len(y_test)))



    ''' find imgs whos prediction is correct and generate error patter'''
    good_ones = np.where(np.all(codewords_1 == label_test_hadamard, axis=1))[0]
    print('good_ones',good_ones)
    select_good_ones = good_ones[200: 300]
    print('select_good_ones',select_good_ones)

    error_patterns = []
    for codeword in codewords_1[select_good_ones]:
        while True:
            error = np.ones(16)
            error[1 + np.random.choice(15, 5, replace=False)] = -1

            if ecoc_model._codeword2label(codeword * error) != ecoc_model._codeword2label(codeword):
                break
        print('clean codewrod:{}\n'
              'target codeword:{}\n'
              'clean label:{}\n'
              'decode target:{}\n'
              .format(codeword, error * codeword, ecoc_model._codeword2label(codeword),
                      ecoc_model._codeword2label(error * codeword)))
        error_patterns.append(error)

    error_patterns = np.array(error_patterns)
    assert error_patterns.shape == (100,16), error_patterns.shape
    np.save(os.path.join(MODEL_DIR, 'mnist/ECOC/3rd_tobe_attacked_mnist_testset_index.npy',
            select_good_ones))
    np.save(os.path.join(MODEL_DIR, 'mnist/ECOC/3rd_tobe_attacked_mnist_testset_error_pattern.npy'),
            error_patterns)
