from keras_vggface import VGGFace
from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow
# from foolbox.models.base import Model
'''
not used anymore, now models are specifically defined according to dataset'''


class LatentModel:
    def __init__8models(self, weights_path_list, model_architecture = 'vgg16', bounds=(0, 1), channel_axis=3):
        """

        :param weights_path_list:
        :param model_architecture: keras model instance that should be used to load pre-trained weights
        """
        # assert weights_path_list.__len__ == 7, 'not 7 model paths provided'
        self._weight_paths = weights_path_list
        print('The model list are considered as in order, which means')
        for idx, path in enumerate(weights_path_list):
            print('Nuber {} model is {}'.format(idx, path))
        self.model = self.build_p_model(weights_path_list)

        self._num_classes = 17
        # number of identities + rejection, the rejection class works as a stop signal to block
        # attacker's further querys, in the code it is simply taken as another class to the
        # attack will stop under mis-classfication criteria, problem remain with targart attack since it
        # would require more modification to foolbox, so not now

        self._bounds = bounds
        self._channel_axis = channel_axis
        self._G = np.array([[1, 0, 0, 0, 1, 1, 1, 0],
                            [0, 1, 0, 0, 1, 1, 0, 1],
                            [0, 0, 1, 0, 1, 0, 1, 1],
                            [0, 0, 0, 1, 0, 1, 1, 1]])
        self._H = np.array([[1, 1, 1, 0, 1, 0, 0, 0],
                            [1, 1, 0, 1, 0, 1, 0, 0],
                            [1, 0, 1, 1, 0, 0, 1, 0],
                            [0, 1, 1, 1, 0, 0, 0, 1]]).T
        self._correctable_sydrome = np.mod(np.dot(np.eye(8), self._H), 2)
        # the index in this mat indicts the error bit
        #  in other word, if sydrome is the first row of this mat
        #  then the first bit is error
        self.CODEBOOK = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]])
        self.__num_illegal_codeword = 0

    def __init__(self, weights_path_list, model_architecture = 'vgg16', bounds=(0, 1), channel_axis=3):
        """

        :param weights_path_list:
        :param model_architecture: keras model instance that should be used to load pre-trained weights
        """
        # assert weights_path_list.__len__ == 7, 'not 7 model paths provided'
        self._weight_paths = weights_path_list
        print('The model list are considered as in order, which means')
        for idx, path in enumerate(weights_path_list):
            print('Nuber {} model is {}'.format(idx, path))
        self.model = self.build_p_model(weights_path_list)
        self._num_classes = 16
        self._bounds = bounds
        self._channel_axis = channel_axis
        self._G = np.array([[1, 0, 0, 0, 1, 1, 1],
                            [0, 1, 0, 0, 1, 1, 0],
                            [0, 0, 1, 0, 1, 0, 1],
                            [0, 0, 0, 1, 0, 1, 1]])
        self._H = np.array([[1, 1, 1, 0, 1, 0, 0],
                            [1, 1, 0, 1, 0, 1, 0],
                            [1, 0, 1, 1, 0, 0, 1]]).T
        self._correctable_sydrome = np.mod(np.dot(np.eye(7), self._H), 2)
        # the index in this mat indicts the error bit
        #  in other word, if sydrome is the first row of this mat
        #  then the first bit is error
        self.CODEBOOK = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        # statistical records
        self.__num_illegal_codeword = 0
        self.__total_prediction_calls = 0

    @classmethod
    def codebook(cls):
        return np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ])

    def report_statistics(self, reset=True):
        if reset:
            print('Before reset\n'
                  'Num of illegal calls={}\n'
                  'Num of totall calls={}\n'.format(self.__num_illegal_codeword, self.__total_prediction_calls))
            self.reset()
        print('After reset\n'
              'Num of illegal calls={}\n'
              'Num of totall calls={}\n'.format(self.__num_illegal_codeword, self.__total_prediction_calls))
        return self.num_illegal_codeword, self.num_total_calls

    def reset(self):
        self.__num_illegal_codeword = 0
        self.__total_prediction_calls = 0

    def build_p_model(self, weights_path_list):

        input = Input((224, 224, 3))

        bottom1 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten1 = Flatten()(bottom1.output)
        dense1 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten1)
        model1 = Model(inputs=input, outputs=dense1)
        model1.load_weights(weights_path_list[0])
        for layer in model1.layers:
            layer.name += '_1'

        bottom2 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten2 = Flatten()(bottom2.output)
        dense2 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten2)
        model2 = Model(inputs=input, outputs=dense2)
        model2.load_weights(weights_path_list[1])
        for layer in model2.layers:
            layer.name += '_2'

        bottom3 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten3 = Flatten()(bottom3.output)
        dense3 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten3)
        model3 = Model(inputs=input, outputs=dense3)
        model3.load_weights(weights_path_list[2])
        for layer in model3.layers:
            layer.name += '_3'

        bottom4 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten4 = Flatten()(bottom4.output)
        dense4 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten4)
        model4 = Model(inputs=input, outputs=dense4)
        model4.load_weights(weights_path_list[3])
        for layer in model4.layers:
            layer.name += '_4'

        bottom5 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten5 = Flatten()(bottom5.output)
        dense5 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten5)
        model5 = Model(inputs=input, outputs=dense5)
        model5.load_weights(weights_path_list[4])
        for layer in model5.layers:
            layer.name += '_5'

        bottom6 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten6 = Flatten()(bottom6.output)
        dense6 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten6)
        model6 = Model(inputs=input, outputs=dense6)
        model6.load_weights(weights_path_list[5])
        for layer in model6.layers:
            layer.name += '_6'

        bottom7 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3),
                          input_tensor=input)
        flatten7 = Flatten()(bottom7.output)
        dense7 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten7)
        model7 = Model(inputs=input, outputs=dense7)
        model7.load_weights(weights_path_list[6])
        for layer in model7.layers:
            layer.name += '_7'

        # bottom8 = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3)
        # ,input_tensor= input)
        # flatten8 = Flatten()(bottom8)
        # dense8 = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten8)
        # model8 = Model(inputs=bottom8.input, outputs=dense8)
        # model8.load_weights(weights_path_list[7])
        # for layer in model8.layers:
        #     layer.name += '_8'

        p_model = Model(inputs=input,
                        outputs=Concatenate()(
                            [model1.layers[-1].output, model2.layers[-1].output, model3.layers[-1].output,
                             model4.layers[-1].output, model5.layers[-1].output, model6.layers[-1].output,
                             model7.layers[-1].output]))
        return p_model

    def reset_num_illegal(self, num=0):
        if num == 0:
            self.__num_illegal_codeword = 0
        else:
            self.__num_illegal_codeword += num

    @property
    def num_total_calls(self):
        return self.__total_prediction_calls

    @property
    def num_illegal_codeword(self):
        return self.__num_illegal_codeword

    def num_classes(self):
        return self._num_classes

    def nb_classes(self):  # for art
        return self._num_classes

    # for art
    @property
    def channel_index(self):
        return self._channel_axis

    @property  # for art
    def input_shape(self):
        return self.model.input_shape[1:]

    def bounds(self):
        return self._bounds

    @property  # for art models
    def clip_values(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    def _decoding(self, r):
        sydrome = np.mod(np.dot(r, self._H), 2)
        if (sydrome == np.array([0, 0, 0])).all():
            return r
        elif (sydrome == self._correctable_sydrome).all(1).any():
            error_pattern = np.squeeze(1 * (self._correctable_sydrome == sydrome).all(1))
            return np.mod(r + error_pattern, 2)
        else:
            self.__num_illegal_codeword += 1
            return -7 * np.ones(r.shape)

    def _decoding_batch(self,rs):
        """

        :param rs: array of predict word
        :return: array of decode word, with error detectons indicates as int -7
        """

        sydromes = np.mod(np.dot(rs, self._H), 2)
        # noerrors = (sydromes == np.array([0,0,0,0])).all(1)
        correctables = np.array([np.squeeze(1*(self._correctable_sydrome == s).all(1)) if s in self._correctable_sydrome
                                else np.zeros(7) for s in sydromes])
        # error patterns if correctable sydrome, or Zeros, which means either no error or un correctable
        # should be directly add to received codeword
        dcs = np.mod(rs + correctables, 2)
        new_sydromes = np.mod(np.dot(dcs, self._H),2)
        EAs = (new_sydromes !=0).any(1) # num of True in EAs should be num of detected E/As
        self.__num_illegal_codeword += list(EAs).count(True)

        dcs1 = np.array(dcs, dtype=np.unicode)
        dcs1[EAs]='E/A' # candidate for other possible way of handeling Error or Adv
        dcs[EAs] = -7
        return dcs

    def _decoding_without_count(self, r):
        sydrome = np.mod(np.dot(r, self._H), 2)
        if (sydrome == np.array([0, 0, 0])).all():
            return r
        elif (sydrome == self._correctable_sydrome).all(1).any():
            error_pattern = np.squeeze(1 * (self._correctable_sydrome == sydrome).all(1))
            return np.mod(r + error_pattern, 2)
        else:
            # self.__num_illegal_codeword += 1
            return -7 * np.ones(r.shape)

    def _decoding_batch_without_count(self, rs):
        """

        :param rs: array of predict word
        :return: array of decode word, with error detectons indicates as int -7
        """

        sydromes = np.mod(np.dot(rs, self._H), 2)
        # noerrors = (sydromes == np.array([0,0,0,0])).all(1)
        correctables = [np.squeeze(1 * (self._correctable_sydrome == s).all(1)) if s in self._correctable_sydrome
                        else np.zeros(7) for s in sydromes]
        # error patterns if correctable sydrome, or Zeros, which means either no error or un correctable
        # should be directly add to received codeword
        dcs = np.mod(rs + correctables, 2)
        new_sydromes = np.mod(np.dot(dcs, self._H), 2)
        EAs = (new_sydromes != 0).any(1)  # num of True in EAs should be num of detected E/As
        # self.__num_illegal_codeword += list(EAs).count(True)

        dcs1 = np.array(dcs, dtype=np.unicode)
        dcs1[EAs] = 'E/A'  # candidate for other possible way of handeling Error or Adv
        dcs[EAs] = -7
        return dcs

    def _codeword2label(self, codeword):
        '''

        :param codeword: array or list or tuple
        :return: np.array label
        '''
        if codeword.ndim == 1:
            l = np.array([np.sum(np.array([2**3,2**2,2,1])* np.array(codeword))])
            if l not in list(range(16)):
                return 16
            else:
                return int(l)
        else:
            ls = np.array(np.sum(np.array([2**3,2**2,2,1])* np.array(codeword),axis=1))
            ls = np.where((ls >15) + (ls < 0), 16, ls)
            return ls

    # todo test new predictions
    # old version for latent model that takes 7 inputs
    def predictions(self,image, count_illegal = True):
        if image.ndim != 4:
            image = image[np.newaxis,:,:,:]
        inputs = [image]*7
        predictions = self.model.predict(inputs)
        # predict_codeword = (k1, k2, k3, k4, r1, r2, r3, rp) = predictions.reshape(8, 2)
        predict_codeword = np.argmax(predictions.reshape(7,2),axis=1)
        if count_illegal:
            decoded = self._decoding(np.array(predict_codeword))
        else:
            decoded =self._decoding_without_count(np.array(predict_codeword))
        if (decoded >15).any() or (decoded<0).any():
            # print('error/attack dectected! ')
            return to_categorical(16)
        else:
            return to_categorical(self._codeword2label(decoded[:4]))

    # old version for latent model that takes 7 inputs
    def batch_predictions(self, images, count_illegal = True):
        inputs = [images]*7
        # predict_codewords = self.model.predict(inputs)
        predictions = self.model.predict(inputs)
        assert predictions.shape == (images.shape[0], 14) # shoule be ( num img, 2 * num model)
        predict_codewords = np.argmax(predictions.reshape(images.shape[0],7,2), axis=2)

        if count_illegal:
            decoded = self._decoding_batch(predict_codewords)
        else:
            decoded = self._decoding_batch_without_count(predict_codewords)
        return to_categorical(self._codeword2label(decoded[:, :4]))
        # do not load all 7 models all together for now
    # # methods should be written rather carefully, to cover all what the attack need
    # def old_predictions(self,image):
    #     predict_code = []
    #     for path in self._weight_paths:
    #         self.model.load_weights(path)
    #         predict_code.append(np.argmax(self.model.predict(image[np.newaxis,::]))) # image bound should be (0,1)
    #     decoded = self._decoding(np.array(predict_code))
    #     if (decoded >15).any() or (decoded<0).any():
    #         # print('error/attack dectected! ')
    #         return to_categorical(16)
    #     else:
    #         return to_categorical(self._codeword2label(decoded[:4]))
    #
    # def old_batch_predictions(self,images):
    #     assert images.ndim > 1
    #     _p = []
    #     for path in self._weight_paths:
    #         self.model.load_weights(path)
    #         bitis = np.argmax(self.model.predict_on_batch(images),axis=1)
    #         # predict_on_batch should return arrays of predictions
    #         _p.append(bitis)
    #     predict_words = np.array(_p).T
    #     # now predict_words should be an array whos row is one codeword prediction of one image
    #     decoded = self._decoding_batch(predict_words)
    #     return to_categorical(self._codeword2label(decoded[:,:4]))
    #     # return softmax(category)
    #     # first consider a close set classification, so error correction should be considered
    #
    # def old_predictions_without_count(self,image):
    #     predict_code = []
    #     for path in self._weight_paths:
    #         self.model.load_weights(path)
    #         predict_code.append(np.argmax(self.model.predict(image[np.newaxis,::]))) # image bound should be (0,1)
    #     decoded = self._decoding_without_count(np.array(predict_code))
    #     if (decoded >15).any() or (decoded<0).any():
    #         print('error/attack dectected! ')
    #         return to_categorical(16)
    #     else:
    #         return to_categorical(self._codeword2label(decoded[:4]))
    #
    # def old_batch_predictions_without_count(self,images):
    #     assert images.ndim >1
    #     _p = []
    #     for path in self._weight_paths:
    #         self.model.load_weights(path)
    #         bitis = np.argmax(self.model.predict_on_batch(images),axis=1)
    #         # predict_on_batch should return arrays of predictions
    #         _p.append(bitis)
    #     predict_words = np.array(_p).T
    #     # now predict_words should be an array whos row is one codeword prediction of one image
    #     decoded = self._decoding_batch_without_count(predict_words)
    #     return to_categorical(self._codeword2label(decoded[:,:4]))
    # cover art requires
    def predict(self, images, batch_size = None, output_type = 'one-hot', count_illegal = True):
        # todo logits not implement yet
        assert output_type in ['one-hot', 'hard_label', 'probabilities', 'codewords']
        if images.ndim == 3:
            images = images[np.newaxis]
            assert images.ndim == 4
        self.__total_prediction_calls += images.shape[0]
        # print('total calls now: {}'.format(self.__total_prediction_calls))
        predictions = self.model.predict(images) # shape = (n,14) # probabilities
        assert predictions.ndim == 2
        if output_type == 'probabilities':
            return predictions

        if predictions.shape[0] > 1:
            # multi images
            raw_codewords = np.argmax(predictions.reshape(images.shape[0], 7, 2), axis=2)
            if count_illegal:
                decoded = self._decoding_batch(raw_codewords)
            else:
                decoded = self._decoding_batch_without_count(raw_codewords)
            hard_label = self._codeword2label(decoded[:, :4])

            if output_type == 'codewords':
                return decoded
            elif output_type == 'hard_label':
                return hard_label
            elif output_type == 'one-hot':
                return to_categorical(hard_label, num_classes=self.num_classes())
        else:
            # single images
            raw_codewords = np.argmax(predictions.reshape(7,2), axis=1)
            if count_illegal:
                decoded = self._decoding(raw_codewords)
            else:
                decoded = self._decoding_without_count(raw_codewords)
            hard_label = self._codeword2label(decoded[:4])

            if output_type == 'codewords':
                return decoded
            elif output_type == 'hard_label':
                return hard_label
            elif output_type == 'one-hot':
                return to_categorical(hard_label, num_classes=self.num_classes())


def vggface2_16classes_resnet50(model_weights = None):
    ''' model prepare'''
    bottom = VGGFace(include_top=False, model='resnet50', weights='vggface', input_shape=(224, 224, 3))
    flatten = Flatten()(bottom.output)
    dense = Dense(units=16, kernel_initializer="he_normal", activation='softmax')(flatten)
    model = Model(inputs=bottom.input, outputs=dense)
    if model_weights:
        model.load_weights(model_weights)
    return model

def vggface2_16classes_src_VGG16(model_weights=None):
    ''' model prepare'''
    bottom = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3))
    flatten = Flatten()(bottom.output)
    dense = Dense(units=16, kernel_initializer="he_normal", activation='softmax')(flatten)
    model = Model(inputs=bottom.input, outputs=dense)
    if model_weights:
        model.load_weights(model_weights)
    return model

def vggface2_2classes_base_VGG16(model_weights=None):
    ''' model prepare'''
    bottom = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3))
    flatten = Flatten()(bottom.output)
    dense = Dense(units=2, kernel_initializer="he_normal", activation='softmax')(flatten)
    model = Model(inputs=bottom.input, outputs=dense)
    if model_weights:
        model.load_weights(model_weights)
    return model

def get_binary_relevance_model():
    input_image = tensorflow.keras.layers.Input(shape=(224, 224, 3))
    # Block 1
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tensorflow.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = tensorflow.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tensorflow.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tensorflow.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = tensorflow.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = tensorflow.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = tensorflow.keras.layers.Flatten()(x)
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    #
    # output = layers.Dense(20, name='dense1')(x)
    # output = layers.Activation('sigmoid', name='output_layer')(output)

    x = tensorflow.keras.layers.Dense(128, activation='relu')(x)
    x = tensorflow.keras.layers.Dropout(0.3)(x)
    x = tensorflow.keras.layers.Dense(128, activation='relu')(x)
    x = tensorflow.keras.layers.Dropout(0.3)(x)

    output = tensorflow.keras.layers.Dense(1, name='dense1')(x)
    output = tensorflow.keras.layers.Activation('sigmoid', name='output_layer')(output)

    model = tensorflow.keras.models.Model(input_image, output)

    return model