""" networks structures that used """
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input, Concatenate, Conv2D, Activation, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras
import numpy as np

def get_1bit_ensemble_vgg16(model_weights_path=None, input_tensor=None):
    bottom = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), input_tensor=input_tensor)
    flatten = Flatten()(bottom.output)
    fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
    output = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(fc2)
    model = Model(inputs=bottom.input, outputs=output)
    if model_weights_path:
        model.load_weights(model_weights_path)
    return model

def get_32class_surrogate_model(model_weights_path=None, input_tensor=None):
    bottom = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), input_tensor=input_tensor)
    flatten = Flatten()(bottom.output)
    fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
    output = Dense(units=32, kernel_initializer="he_normal", activation='softmax', name='output')(fc2)
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
        self.__hadamard_matrix = hadamard_matrix # ndarray

    # properties
    @property
    def keras_model(self):
        return self.__keras_model
    @property
    def hadamard_matrix(self):
        return self.__hadamard_matrix
    @property
    def num_class(self):
        return self.keras_model.output_shape


        # build model
    def build_chuan_model(self, bit_model_weight_path_in_order):
        # Chinese character 川 that all models works independently but sharing one same input image
        # base models been stored into a list, needs test
        input = Input((64, 64, 3), name='very_input')

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
            bottom = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), input_tensor=input_tensor)
            flatten = Flatten()(bottom.output)
            fc1 = Dense(units=4096, activation='relu', name='fc1')(flatten)
            fc2 = Dense(units=4096, activation='relu', name='fc2')(fc1)
            output = Dense(units=1, kernel_initializer="he_normal", activation='tanh', name='output')(fc2)
            model = Model(inputs=bottom.input, outputs=output)
            if full_model_weight_path:
                model.load_weights(full_model_weight_path)
            else:
                model.load_weights(
                        'Model/gtsrb/model_weight/Hadamard32_surrogate_weights_freeze6_bit_1/final_trained_weights.hdf5')
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
        very_input = Input((64, 64, 3), name='very_input')
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
            for layer in bit_branch.layers:
                connected = layer(connected)
            reconnect_branch_ends.append(connected)

        cat_model = Model(inputs=very_input,
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
        hinge_losses = [tensorflow.keras.losses.hinge(y_pred=y_pred, y_true=y_trues[i]) for i, y_pred in enumerate(individual_activations)]
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
        dp_logits = np.matmul(outputs, self.hadamard_matrix.T)  # shape: (num_img, 32) * (32, 32), hadamard matrix is Symmetric
        # for tanh activation function, one relu effect, probably needs consider other activation func in the future

        if output_type in ['dot_product_logits', 'ecoc_logits']:
            return dp_logits

        codeword = self.hadamard_matrix[np.argmax(dp_logits, axis=dp_logits.ndim-1)]
        # todo correlation decoding is not the best, thus index label accuracy is not so good, add one minimum distance decoding
        if output_type in ['codewords' , 'codeword' ,'decoded']:
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
            md_codeword = self.hadamard_matrix[index_labels]
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

class Wrapped_Keras_Model():
    def __init__(self, keras_model):
        self.keras_model = keras_model
    def predict(self,img, output_type='probabilities'):
        if img.ndim == 3:
            img = img[np.newaxis]
        predictions = self.keras_model.predict(img)
        if output_type in ['probability', 'probabilities']:
            return predictions
        if output_type in ['label', 'hard_label', 'softmax_label']:
            return np.argmax(predictions, axis=-1)
