


""" the tensorflow version is 2.0.0/2.1.0, in theory the most of tensorflow versions should work well as long as the
eager execution mode is deactivated, but as I use symbolic funtion of Keras, you propobably shouldn't use the tensorflow
of too late version because new Keras has abandoned symbolic function """
import time

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
# import keras
import tensorflow.keras.backend as K
import numpy as np
import math
# from bch_gtsrb.models import BCHn15t3Model
# from ecoc_hadamard_gtsrb.models import ECOC_Hadamard_Model
from tensorflow.keras.models import Model as tfkerasModel
from tensorflow.keras.models import Model
from collections import namedtuple
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from core_attack.adversarial import LatentAdversarial, BetterAdversarial, UltimateAdversarial, \
    UltimateAdversarial_multilabel
from utils.math_func import compute_psnr
from scipy import optimize
from scipy.special import softmax
from cvxopt import matrix
from cvxopt import solvers
import tensorflow.keras


""" 
The ECOCAttack class is a dummy big class for ECOC model, it is only used in the ECOC paper, in this class, the __init__
method is used for providing things need for each attack, like keras function of computing logits and gradients. Other
methods like CWBattack, CW_correlation is the implementation of each Attack. As the structure of this ECOCAttack is pretty
dummy, this class is abandoned shortly after we came up with the final version of the Jacobian Attack.
"""
class ECOCAttack(object):

    def __init__(self, ecoc_model):
        if hasattr(ecoc_model, 'keras_model'):
            self.model = ecoc_model
            keras_model = ecoc_model.keras_model
            # keras model is just for backend opreation, while LatentModel instance is for prediction and other stuff
            assert isinstance(keras_model, tf.keras.models.Model)
        else:
            raise TypeError('Input model must be ethier customed wrapped model')

        # activated output level
        _y = K.reshape(keras_model.output[0],[-1])
        _output_list = tf.unstack(_y)
        Jacbs_list = [K.gradients(yi, keras_model.input) for yi in _output_list]
        Jacbs = tf.stack(Jacbs_list) # after stack, the shape strangely becomes (32, 1,?,64,64,3) dont know if problem
        self._Jacbian_fn = K.function([keras_model.input], [Jacbs])

        # get logits before activation (tanh)
        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self._logits_func = K.function([keras_model.input], logits)
        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, keras_model.input) for li in logits]
        Jacbs_logits = tf.stack(Jacbs_logits_list, name='Jacb_logits')
        self._Jacbian_logits_fn = K.function([keras_model.input],[Jacbs_logits])
        # can be used to valid gradient vanished at activation level
        logits_grad = K.gradients(logits, keras_model.input)
        self._logits_grad_fn = K.function([keras_model.input], logits_grad)
        dodl = K.gradients(keras_model.output, logits)
        self._dodl_grad_fn = K.function([keras_model.input], dodl)

        # the bit models are trained using hinge losses function
        # y_trues serves as both ground truth and attacking target label
        y_trues = K.placeholder(shape=keras_model.output_shape[1])
        hinge_losses = [keras.losses.hinge(y_pred=y_pred, y_true=y_trues[i]) for i, y_pred in
                        enumerate(individual_activations)]
        self._hinge_losses_func = K.function([keras_model.input, y_trues], hinge_losses)

        # confidence loss
        confidence_thres = K.placeholder(shape=(None, 1))
        # thres been expand with a new axis to be aglined with logits, do this when call gradient func
        confidence_losses = K.min([z * 2 * y_trues[i] for i, z in enumerate(logits)] + [confidence_thres])
        self._confidence_losses_func = K.function([keras_model.input, y_trues, confidence_thres], [confidence_losses])
        dcdx = K.gradients(confidence_losses, keras_model.input)  # derivative of confidence over image input x
        self._confidence_gradient_func = K.function([keras_model.input, y_trues, confidence_thres], dcdx)

        # eoco logits
        hadamard = K.variable(self.model.hadamard_matrix.T)
        ecoc_logits = K.dot(keras_model.output, hadamard) # shape: (num_img, 32) * (32, 32), hadamard matrix is Symmetric
        self._ecoc_logits_func = K.function([keras_model.input], [ecoc_logits])



        # takes indexes to determinate target and second top class
        class_maximize = K.placeholder(dtype='int32')
        class_minimize = K.placeholder(dtype='int32')
        ecoc_confidence_losses = K.max([ecoc_logits[:,class_minimize] - ecoc_logits[:,class_maximize], -confidence_thres[:,0]])
        self._ecoc_confidence_loss_func = K.function([keras_model.input, class_minimize, class_maximize, confidence_thres], [ecoc_confidence_losses])
        ecoc_confidence_grad = K.gradients(ecoc_confidence_losses, keras_model.input)
        self._ecoc_confidence_grad_fn = K.function([keras_model.input, class_minimize, class_maximize, confidence_thres], ecoc_confidence_grad)

        # LOTS
        target_t = K.placeholder(shape=(1,len(logits)))
        logits_tensor = K.concatenate(logits)
        LOTS_object = K.dot(target_t - logits_tensor, K.transpose(target_t - logits_tensor)) / 2
        LOTS_object_L2 = K.sum(K.square(target_t - logits_tensor)) / 2
        LOTS_grad = K.gradients(LOTS_object, keras_model.input)
        LOTS_grad_L2 = K.gradients(LOTS_object_L2, keras_model.input)
        self._LOTS_grad_func = K.function([keras_model.input, target_t], LOTS_grad)
        self._LOTS_grad_func_L2 = K.function([keras_model.input, target_t], LOTS_grad_L2)

    def CWBattack(self, clean_img, confidence=0.0, initial_constant=0.01, learning_rate=5e-3,
                  num_binary_search=10, max_iterations=1000, targeting_un_decoded=None):

        """
        This is the BowenAttack that we proposed in the ECOC paper

        :param clean_img:
        :param confidence:
        :param initial_constant:
        :param learning_rate:
        :param num_binary_search:
        :param max_iterations:
        :param targeting_un_decoded: {-1, +1} obtained directly by sign function
        :return:
        """
        ecoc_CWB_adv = UltimateAdversarial(clean_img, is_target_attack=True, target_output=targeting_un_decoded,
                                               latent_model=self.model, better_criterion='psnr', output_type='decoded',
                                           conf_thres = confidence, strict_conf_thres = True) # really careful
        # variables will be used for whole attack
        clean_code = ecoc_CWB_adv.clean_output
        min_, max_ = (0, 1)
        target_error = ecoc_CWB_adv.target_error
        # adam algorithm for gradient updating
        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_img.shape)

        # target loss should be aligned with target code, i.e. using 1 indicates those unchanged label and -1 those changed
        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999 #0.9999999999999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(w):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            w = np.tanh(w)

            grad = 1 - np.square(w)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            w = w * b + a

            grad = grad * b
            return w, grad

        def overall_loss_grad(img, reconstruct_clean_img, raw_tar_code, constant, confidence):
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty

            confidence = np.array(confidence)
            while confidence.ndim < 2:
                confidence = confidence[np.newaxis]

            confidence_loss = self._confidence_losses_func([img, raw_tar_code, confidence])[
                0]  # this loss should be maxmized to get desired error
            # print('confidence loss:', confidence_loss)

            confidence_grad = self._confidence_gradient_func([img, raw_tar_code, confidence])[0]

            # L2 distance loss, grad w.r.t. img
            squared_l2_distance = np.sum((img - reconstruct_clean_img) ** 2) / s ** 2
            squared_l2_distance_grad = (2 / s ** 2) * (img - reconstruct_clean_img)

            # total loss, grad w.r.t. img
            overall_loss = squared_l2_distance - constant * confidence_loss
            overall_grad = squared_l2_distance_grad - constant * confidence_grad
            return overall_loss, overall_grad

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_clean_img = to_attack_space(clean_img)
        # will be close but not identical to clean_img
        reconstructed_clean_img, _ = to_model_space(att_clean_img)

        # binary search for good constant c
        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_constant
        lower_bound = 0
        upper_bound = np.inf
        for binary_search_step in range(num_binary_search):
            att_perturbation = np.zeros_like(att_clean_img)
            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf
            for iteration in range(max_iterations):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)
                # compute current confidence
                current_logits = np.squeeze(np.array(self._logits_func([perturbed])))
                bit_confidences = 2* targeting_un_decoded * current_logits
                current_confidence = np.min(bit_confidences)

                is_adv, linf, psnr = ecoc_CWB_adv.new_perturbed(perturbed=perturbed, constant=const,
                                                                num_iteration=iteration, conf=current_confidence,
                                                                num_binary_search = binary_search_step)
                # is_adv, linf, per_predictions, per_predict_codeword = self._is_adv(reconstructed_clean_img, perturbed,
                #                                                                    target_code)
                model_space_loss, model_space_grad = overall_loss_grad(perturbed, reconstructed_clean_img,
                                                                       targeting_un_decoded,
                                                                       const, confidence)
                # print('total_loss in model space = %.f', model_space_loss)
                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grad * dpertub_dw
                gradient_vanished = True if np.sum(np.square(att_space_grad)) == 0 else False
                if gradient_vanished == True:
                    ecoc_CWB_adv.update_records(gradient_vanished=True)
                    assert np.sum(np.square(optimizer(att_space_grad, learning_rate))) == 0
                    break
                att_perturbation -= optimizer(att_space_grad, learning_rate)


                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True
                    # break # for saving time

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    # _activated = self.model.predict(perturbed, output_type='activated')
                    # print(_activated)
                    # print('is adv = {}, adversarial.idv_found = {}'
                    #       .format(np.array_equal(np.sign(_activated), targeting_un_decoded),
                    #               ecoc_CWB_adv.best_adv is not None)                          )
                    # print('model space loss = {}'.format(model_space_loss))
                    # after each tenth of the iterations, check progress
                    if not (model_space_loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = model_space_loss

            # binary search part
            if found_adv:
                # logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                # logging.info(
                #     "failed to find adversarial " "with const = {}".format(const)
                # )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

        return ecoc_CWB_adv

    def CW_correlation(self, clean_img, confidence=0.0, initial_constant=0.01, learning_rate=5e-3,
                       num_binary_search=10, max_iterations=1000, targeting_ecoc_logits_num=None):

        """
        This is my implementation of C&W attack for the ECOC model

        :param clean_img:
        :param confidence:
        :param initial_constant:
        :param learning_rate:
        :param num_binary_search:
        :param max_iterations:
        :param targeting_ecoc_logits_num: targeting "class" by sorting the ecoc logits, same as ground label is non-target attack
        :return:
        """
        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(w):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            w = np.tanh(w)

            grad = 1 - np.square(w)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            w = w * b + a

            grad = grad * b
            return w, grad

        def overall_loss_grad(img, reconstruct_clean_img, class_minimize, class_maximize, constant, confidence):
            """

            :param img:
            :param reconstruct_clean_img:
            :param who_minus_whom: # e.g. [ 0, 0, 1, -1, 0, 0]  indicates the third minus the forth
            :param constant:
            :param confidence:
            :return:
            """
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty
            # gradient_vanished = False
            confidence = np.array(confidence)
            while confidence.ndim < 2:
                confidence = confidence[np.newaxis]

            confidence_loss = self._ecoc_confidence_loss_func([img, class_minimize, class_maximize, confidence])[
                0]  # this loss should be maxmized to get desired error

            confidence_grad = self._ecoc_confidence_grad_fn([img, class_minimize, class_maximize, confidence])[0]

            # L2 distance loss, grad w.r.t. img
            squared_l2_distance = np.sum((img - reconstruct_clean_img) ** 2) / s ** 2
            squared_l2_distance_grad = (2 / s ** 2) * (img - reconstruct_clean_img)

            # total loss, grad w.r.t. img
            overall_loss = squared_l2_distance + constant * confidence_loss
            overall_grad = squared_l2_distance_grad + constant * confidence_grad
            return overall_loss, overall_grad

        # def compute_confidence(img, class_minimize, class_maximize):
        #
        #     return  ecoc_logits[:,class_minimize] - ecoc_logits[:,class_maximize]

        is_targeted = False if targeting_ecoc_logits_num == None else True
        print('the target output input to Adversarial object is ', targeting_ecoc_logits_num)
        ecoc_CW_adv = UltimateAdversarial(clean_img, is_target_attack=is_targeted, target_output=targeting_ecoc_logits_num,
                                        latent_model=self.model, better_criterion='psnr', output_type='ecoc_hard_label',
                                          conf_thres = confidence, strict_conf_thres = True)

        # variables will be used for whole attack
        clean_output = ecoc_CW_adv.clean_output
        min_, max_ = (0, 1)
        target_error = ecoc_CW_adv.target_error
        # adam algorithm for gradient updating
        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_img.shape)

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_clean_img = to_attack_space(clean_img)
        # will be close but not identical to clean_img
        reconstructed_clean_img, _ = to_model_space(att_clean_img)

        # binary search for good constant c
        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_constant
        lower_bound = 0
        upper_bound = np.inf
        for binary_search_step in range(num_binary_search):
            att_perturbation = np.zeros_like(att_clean_img)
            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf
            for iteration in range(max_iterations):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)

                # do one forward to get ecoc logits and find how who should minus whom
                ecoc_logits = np.squeeze(self._ecoc_logits_func([perturbed])[0])
                if ecoc_CW_adv.target_output is None:
                    # print('========HUGE WARNING=======\nThis supposed to be a target-attack script (for now)')
                    class_minimize = int(ecoc_CW_adv.clean_output)
                    x = np.zeros_like(ecoc_logits)
                    x[class_minimize] = np.inf
                    class_maximize = np.argmax(ecoc_logits - x)
                else:
                    class_maximize = targeting_ecoc_logits_num
                    x = np.zeros_like(ecoc_logits)
                    x[targeting_ecoc_logits_num] = np.inf
                    class_minimize = np.argmax(ecoc_logits-x)

                current_confidence = ecoc_logits[class_maximize] - ecoc_logits[class_minimize]

                is_adv, linf, psnr = ecoc_CW_adv.new_perturbed(perturbed=perturbed, constant=const,
                                                                num_iteration=iteration, conf = current_confidence)

                model_space_loss, model_space_grad = overall_loss_grad(perturbed, reconstructed_clean_img,
                                                                       class_minimize, class_maximize,
                                                                       const, confidence)
                # ecoc_logits[class_minimize] - ecoc_logits[class_maximize] == model_space_loss should be true

                # print('total_loss in model space = %.f', model_space_loss)
                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grad * dpertub_dw

                gradient_vanished = True if np.sum(att_space_grad) == 0 else False

                if gradient_vanished == True:
                    ecoc_CW_adv.update_records(gradient_vanished=True)
                    assert np.sum(optimizer(att_space_grad, learning_rate)) == 0
                    break
                att_perturbation -= optimizer(att_space_grad, learning_rate)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    # print('activated output', self.model.predict(perturbed, output_type='activated'))
                    # print('ECOC logits',ecoc_logits)
                    # print('model space loss', model_space_loss)

                    # after each tenth of the iterations, check progress
                    if not (model_space_loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = model_space_loss

            # binary search part
            if found_adv:
                # logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                # logging.info(
                #     "failed to find adversarial " "with const = {}".format(const)
                # )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

        return ecoc_CW_adv

    def LogitLOTSAttack(self, clean_img, target_imgs, targeting_codeword, **kwargs):
        """
        My implementation of the LOTS attack at logits level, for ECOC model

        :param perturbed: single to-be attacked image
        :param target_imgs: some images belong to the same class that adversary intend to let the adv image falls into, better be list
        :param targeting_codeword: aligned with target_imgs since target_imgs should be correctly classified
        :param kwargs:  max_iteration, no step_size since not normalized
        :return:
        """
        # clean_logits = self.model.predict(clean_img, output_type='logits')

        class LOSTadv(UltimateAdversarial):
            def __init__(self, clean_img, latent_model=None, is_target_attack=True, target_output = None, clean_img_path=None,
                 better_criterion='psnr', output_type = False, conf_thres = 0.0, strict_conf_thres = True):
                super(LOSTadv, self).__init__(clean_img, latent_model=latent_model, is_target_attack=is_target_attack, target_output = target_output, clean_img_path=clean_img_path,
                 better_criterion=better_criterion, output_type = output_type, conf_thres = conf_thres, strict_conf_thres = strict_conf_thres)
                self.__best_distance = super(LOSTadv, self).best_distance
                self.__best_adv = super(LOSTadv, self).best_adv
            @property
            def best_adv(self):
                return self.__best_adv
            @property
            def best_distance(self):
                return self.__best_distance
            def new_perturbed(self, perturbed, output_type=None, **kwargs):
                # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
                if output_type is None:
                    output_type = self.__output_type

                is_adv_fn = self.is_target_adv if self.is_target_attack else self.is_non_target_adv
                predict_output = self.model.predict(perturbed, output_type=output_type)
                # print('self.__model.predict(perturbed, output_type=output_type)', predict_output)

                is_adv = is_adv_fn(predict_output)

                linf = np.max(np.abs(self.clean_img - perturbed))
                psnr = compute_psnr(self.clean_img,
                                    perturbed)  #
                if self.better_adv_criterion == 'psnr':
                    new_distance = psnr
                    if is_adv :
                        self.__best_distance = new_distance
                        self.__best_adv = perturbed
                        self.update_records(**kwargs)
                elif self.better_adv_criterion == 'linf':
                    new_distance = linf
                    if is_adv :
                        self.__best_distance = new_distance
                        self.__best_adv = perturbed
                        self.update_records(**kwargs)

                return is_adv, linf, psnr

        max_iter = kwargs['max_iteration']
        target_logits = np.mean(self._logits_func([target_imgs]), axis=1).reshape(1,len(targeting_codeword))
        assert np.array_equal(self.model.hadamard_matrix[np.argmax(np.matmul(target_logits, self.model.hadamard_matrix.T))],
                              targeting_codeword)

        adv = LOSTadv(clean_img, is_target_attack=True, target_output=targeting_codeword,
                                             latent_model=self.model, better_criterion='psnr', output_type='codewords',
                                             conf_thres=None, strict_conf_thres=False)

        perturbed = clean_img
        for i in range(max_iter):
            # current_logits = np.squeeze(np.array(self._logits_func([perturbed])))

            # object_fuc = 1/2 * ||target_logits - clean_logits||^2
            # clean_logit_gradient = self._logits_grad_fn([perturbed])
            # clean_logit_jacb = self._Jacbian_logits_fn([perturbed])[0]
            # clean_logit_gradient = clean_logit_jacb.reshape(32, np.prod(clean_img.shape))
            # gradient = np.matmul(target_logits, clean_logit_gradient) + np.matmul(current_logits, clean_logit_gradient)
            # gradient = gradient.reshape(clean_img.shape)
            func_gradient = self._LOTS_grad_func([perturbed, target_logits])[0]
            # L2_func_gradient = self._LOTS_grad_func_L2([perturbed, target_logits])[0]
            # print('mse between formalized gradient and func_gradient = {}\n'
            #       'mse between func_gradient and L2 norm func gradient = {}\n\n'.format(np.mean(np.square(func_gradient - gradient)),
            #                                                                    np.mean(np.square(L2_func_gradient - func_gradient))))

            gradient = func_gradient

            # L-inf normlize

            gradient = gradient / np.max(np.abs(gradient))

            perturbed = (np.round(perturbed *255) - gradient)/255
            perturbed = np.clip(perturbed, 0, 1)

            is_adv, linf, psnr = adv.new_perturbed(perturbed, output_type='codeword')
            # print('current predictions:\n'
            #       'logits = {}\n'
            #       'codeword = {}\n'
            #       'class = {}\n\n'
            #       .format(self.model.predict(perturbed, output_type='logit'),
            #               self.model.predict(perturbed, output_type='codeword'),
            #               self.model.predict(perturbed, output_type='hard_label')))
            if is_adv:
                adv.update_records(iteration = i)
                break

        return adv

    # def CW_correlation_considering_non_target_attack(self, clean_img, confidence=0.0, initial_constant=0.01, learning_rate=5e-3,
    #                    num_binary_search=10, max_iterations=1000, targeting_ecoc_logits_num=None):
    #
    #     """
    #
    #     :param clean_img:
    #     :param confidence:
    #     :param initial_constant:
    #     :param learning_rate:
    #     :param num_binary_search:
    #     :param max_iterations:
    #     :param targeting_ecoc_logits_num: targeting "class" by sorting the ecoc logits, same as ground label is non-target attack
    #     :return:
    #     """
    #     def to_attack_space(x):
    #         # map from [min_, max_] to [-1, +1]
    #         a = (min_ + max_) / 2
    #         b = (max_ - min_) / 2
    #         x = (x - a) / b
    #
    #         # from [-1, +1] to approx. (-1, +1)
    #         x = x * 0.999999
    #
    #         # from (-1, +1) to (-inf, +inf)
    #         return np.arctanh(x)
    #
    #     def to_model_space(w):
    #         """Transforms an input from the attack space
    #         to the model space. This transformation and
    #         the returned gradient are elementwise."""
    #
    #         # from (-inf, +inf) to (-1, +1)
    #         w = np.tanh(w)
    #
    #         grad = 1 - np.square(w)
    #
    #         # map from (-1, +1) to (min_, max_)
    #         a = (min_ + max_) / 2
    #         b = (max_ - min_) / 2
    #         w = w * b + a
    #
    #         grad = grad * b
    #         return w, grad
    #
    #     def overall_loss_grad(img, reconstruct_clean_img, class_minimize, class_maximize, constant, confidence):
    #         """
    #
    #         :param img:
    #         :param reconstruct_clean_img:
    #         :param who_minus_whom: # e.g. [ 0, 0, 1, -1, 0, 0]  indicates the third minus the forth
    #         :param constant:
    #         :param confidence:
    #         :return:
    #         """
    #         # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
    #         s = max_ - min_  # should be _max - _min
    #         # clean_img is for distance penalty
    #         # gradient_vanished = False
    #         confidence = np.array(confidence)
    #         while confidence.ndim < 2:
    #             confidence = confidence[np.newaxis]
    #
    #         confidence_loss = self._ecoc_confidence_loss_func([img, class_minimize, class_maximize, confidence])[
    #             0]  # this loss should be maxmized to get desired error
    #
    #         confidence_grad = self._ecoc_confidence_grad_fn([img, class_minimize, class_maximize, confidence])[0]
    #
    #         # L2 distance loss, grad w.r.t. img
    #         squared_l2_distance = np.sum((img - reconstruct_clean_img) ** 2) / s ** 2
    #         squared_l2_distance_grad = (2 / s ** 2) * (img - reconstruct_clean_img)
    #
    #         # total loss, grad w.r.t. img
    #         overall_loss = squared_l2_distance + constant * confidence_loss
    #         overall_grad = squared_l2_distance_grad + constant * confidence_grad
    #
    #         # if gradient vanished because of activation
    #         # if np.sum(overall_grad) == 0:
    #         #     gradient_vanished = True
    #         return overall_loss, overall_grad #, gradient_vanished
    #
    #     # print('---------before adv instancelize---------\n'
    #     #       "self.model.predict(clean_img, output_type='ecoc_hard_label')=", self.model.predict(clean_img, output_type='ecoc_hard_label'))
    #     ecoc_CW_adv = UltimateAdversarial(clean_img, is_target_attack=False, target_output=targeting_ecoc_logits_num,
    #                                     latent_model=self.model, better_criterion='psnr', output_type='ecoc_hard_label')
    #
    #     # variables will be used for whole attack
    #     clean_output = ecoc_CW_adv.clean_output
    #     min_, max_ = (0, 1)
    #     target_error = ecoc_CW_adv.target_error
    #     # adam algorithm for gradient updating
    #     from foolbox.attacks.carlini_wagner import AdamOptimizer
    #     optimizer = AdamOptimizer(clean_img.shape)
    #
    #     # variables representing inputs in attack space will be
    #     # prefixed with att_
    #     att_clean_img = to_attack_space(clean_img)
    #     # will be close but not identical to clean_img
    #     reconstructed_clean_img, _ = to_model_space(att_clean_img)
    #
    #     # binary search for good constant c
    #     # the binary search finds the smallest const for which we
    #     # find an adversarial
    #     const = initial_constant
    #     lower_bound = 0
    #     upper_bound = np.inf
    #     for binary_search_step in range(num_binary_search):
    #         att_perturbation = np.zeros_like(att_clean_img)
    #         found_adv = False  # found adv with the current const
    #         loss_at_previous_check = np.inf
    #         for iteration in range(max_iterations):
    #             # do optimize here
    #             # go back to model space to see if is adv
    #             # perturbed is model space, dpertub_dw is derivative of model space over attack space
    #             perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)
    #             is_adv, linf, psnr = ecoc_CW_adv.new_perturbed(perturbed=perturbed, constant=const,
    #                                                             num_iteration=iteration)
    #
    #             # do one forward to get ecoc logits and find how who should minus whom
    #             ecoc_logits = np.squeeze(self._ecoc_logits_func([perturbed])[0])
    #             if ecoc_CW_adv.target_output is None:
    #                 # todo test non-target attack
    #                 class_minimize = int(ecoc_CW_adv.clean_output)
    #                 x = np.zeros_like(ecoc_logits)
    #                 x[class_minimize] = np.inf
    #                 class_maximize = np.argmax(ecoc_logits - x)
    #             else:
    #                 class_maximize = targeting_ecoc_logits_num
    #                 x = np.zeros_like(ecoc_logits)
    #                 x[targeting_ecoc_logits_num] = np.inf
    #                 class_minimize = np.argmax(ecoc_logits-x)
    #
    #             model_space_loss, model_space_grad = overall_loss_grad(perturbed, reconstructed_clean_img,
    #                                                                    class_minimize, class_maximize,
    #                                                                    const, confidence)
    #             # ecoc_logits[class_minimize] - ecoc_logits[class_maximize] == model_space_loss should be true
    #
    #             # print('total_loss in model space = %.f', model_space_loss)
    #             # final gradient and update by Adam algorithm
    #             att_space_grad = model_space_grad * dpertub_dw
    #
    #             gradient_vanished = True if np.sum(att_space_grad) == 0 else False
    #
    #             if gradient_vanished == True:
    #                 ecoc_CW_adv.update_records(gradient_vanished=True)
    #                 assert np.sum(optimizer(att_space_grad, learning_rate)) == 0
    #                 break
    #             att_perturbation -= optimizer(att_space_grad, learning_rate)
    #
    #             if is_adv:
    #                 # this binary search step can be considered a success
    #                 # but optimization continues to minimize perturbation size
    #                 found_adv = True
    #
    #             if iteration % (np.ceil(max_iterations / 10)) == 0:
    #                 # print('activated output', self.model.predict(perturbed, output_type='activated'))
    #                 # print('ECOC logits',ecoc_logits)
    #                 # print('model space loss', model_space_loss)
    #
    #                 # after each tenth of the iterations, check progress
    #                 if not (model_space_loss <= 0.9999 * loss_at_previous_check):
    #                     break  # stop Adam if there has not been progress
    #                 loss_at_previous_check = model_space_loss
    #
    #         # binary search part
    #         if found_adv:
    #             # logging.info("found adversarial with const = {}".format(const))
    #             upper_bound = const
    #         else:
    #             # logging.info(
    #             #     "failed to find adversarial " "with const = {}".format(const)
    #             # )
    #             lower_bound = const
    #
    #         if upper_bound == np.inf:
    #             # exponential search
    #             const *= 10
    #         else:
    #             # binary search
    #             const = (lower_bound + upper_bound) / 2
    #
    #     return ecoc_CW_adv

    def JacobAttack(self, clean_img, targeting_codeword, step_size, max_iteration, **kwargs):
        """
        This method is only for early tests with Jacobian attack, it should be the very early vesion of the Jacbobian attack,
        and is it abandoned shortly after we come up with the final version of the Jacobian attack

        """
        def new_Jac_direction(target_codeword, logit_P, conf_para = 0):
            '''copy fromm new attak'''
            # target_codeword = targeting_un_decoded
            # logit_P = self.model.predict(perturbed, output_type='logits')
            n = len(self.model.hadamard_matrix[0])

            target_label = np.argmax(
                np.matmul(target_codeword, self.model.hadamard_matrix.T))  # denoted as t in the report
            current_label = np.argmax(np.matmul(logit_P, self.model.hadamard_matrix))
            # assert np.array_equal(self.model.hadamard_matrix[target_label], np.squeeze(self.model.predict(clean_img, output_type='codeword'))) # can be deleted after one test

            t = target_label
            # gtl = lambda t,l: np.matmul(logit_P, self.model.hadamard_matrix[t])-np.matmul(logit_P, self.model.hadamard_matrix[l])
            # store all possible gtl first so the whole mat is a constant regardless of t and l, note gtt is 0
            gtls = [
                np.matmul(logit_P, self.model.hadamard_matrix[t]) - np.matmul(logit_P, self.model.hadamard_matrix[l])
                for l in range(n)]
            gtls_no_t = gtls.copy()
            del gtls_no_t[t]
            gols = [np.matmul(logit_P, self.model.hadamard_matrix[current_label]) - np.matmul(logit_P,
                                                                                            self.model.hadamard_matrix[
                                                                                                l]) for l in range(n)]
            gols_no_o = gols.copy()
            del gols_no_o[current_label]

            # conf = np.max(np.abs(gtls_no_t)/n)
            conf = np.min(np.abs(gols_no_o) / n) * conf_para
            # formal def of inner min part of eq.(17) that should be minimized
            # a_func = lambda at: at**2 + np.sum( [ min(0, at + g / n - conf)**2 for g in gtls_no_t] )
            a_func = lambda at: at ** 2 + np.sum([min(0, at + g / n  -conf) ** 2 for g in gtls_no_t])
            # find optimized a_star
            a_t = optimize.brent(a_func)

            # opt_a = np.zeros(n)

            # opt_a = [min(0, a_t + gtls[i] / n - conf) for i in range(n)]
            opt_a = [min(0, a_t + gtls[i] / n - conf) for i in range(n)]
            opt_a[t] = a_t

            # d_star = np.matmul(opt_a, self.model.hadamard_matrix.T)
            d_star = np.matmul(np.array(opt_a).reshape(32), self.model.hadamard_matrix.T)
            '''copy end'''
            return  d_star
        try:
            conf_para = kwargs['conf_para']
        except KeyError:
            conf_para = 0

        try:
            use_d_star = kwargs['use_d_star']
        except KeyError:
            use_d_star = False

        ecoc_Jacob_adv = UltimateAdversarial(clean_img, is_target_attack=True, target_output=targeting_codeword,
                                             latent_model=self.model, better_criterion='psnr', output_type='codewords',
                                             conf_thres = None, strict_conf_thres = False) # really careful
        clean_codeword = ecoc_Jacob_adv.clean_output

        perturbed = clean_img
        J_ranks = []
        # updating iteratively
        for iter in range(max_iteration):
            # current_logits = self.model.predict(perturbed, output_type='logits') # v1.1
            current_logits = self.model.predict(perturbed, output_type='activated') # v1.0
            if iter%40 == 0:
                print(current_logits)

            direction_d = np.array(targeting_codeword - current_logits)
            if use_d_star:
                current_logits = self.model.predict(perturbed, output_type='logits') # v2.1
                d_star = new_Jac_direction(targeting_codeword, current_logits, conf_para=conf_para)
                direction_d = d_star

            # Jacobian_J = self._Jacbian_fn([perturbed])[0] # v1.0: derivative at activated level

            Jacobian_J = self._Jacbian_logits_fn([perturbed])[0] # v1.1: derivative at logits level

            J_matrix = Jacobian_J.reshape(32, np.prod(clean_img.shape))
            # print(Jacobian_J.shape)
            if np.sum(np.square(Jacobian_J)) == 0: #somehow Jacobian is 0
                # no advs can be found, attack is invalid
                is_adv, linf, psnr = ecoc_Jacob_adv.new_perturbed(perturbed, output_type='codewords', num_iter=iter)
                ecoc_Jacob_adv.update_records(stop='0_Jacb')
                break
            # check the rank of J_matrix
            J_rank = np.linalg.matrix_rank(J_matrix)
            J_ranks.append(J_rank)
            # if J_rank == 32 or J_rank==np.prod(clean_img.shape):
            #     ecoc_Jacob_adv.update_records(full_rank = True)
            # update_direction = np.matmul(Jacobian_J.T, direction_d.T).reshape(clean_img.shape) # TODO: fix matrix indices
            update_direction = np.matmul(J_matrix.T, direction_d.T).reshape(clean_img.shape)
            perturbed = perturbed +  step_size * update_direction / np.sqrt(np.sum(np.square(update_direction)))
            perturbed = np.clip(perturbed, 0, 1)

            is_adv, linf, psnr = ecoc_Jacob_adv.new_perturbed(perturbed, output_type = 'codewords', num_iter=iter)
        ecoc_Jacob_adv.update_records(J_ranks=tuple(J_ranks))

        return  ecoc_Jacob_adv

    def NewJacAttack_v1(self, clean_img, target_codeword, confidence = 31):
        """
        Before we came up with the final version of the Jacobian attack, we tried many temp version, this is one of the
        temp versions.
        """

        ecoc_NewJacob_adv = UltimateAdversarial(clean_img, is_target_attack=True, target_output=target_codeword,
                                               latent_model=self.model, better_criterion='psnr', output_type='codewords',
                                           conf_thres = None, strict_conf_thres = False) # really careful
        # logit_P = self.model.predict(clean_img,output_type='logits')  # network output after sigmoid before others
        logit_P = self.model.predict(clean_img, output_type='activated')  # v2.0
        n = len(self.model.hadamard_matrix[0])

        target_label = np.argmax(np.matmul(target_codeword, self.model.hadamard_matrix.T)) # denoted as t in the report
        clean_label = np.argmax(np.matmul(logit_P, self.model.hadamard_matrix))
        # assert np.array_equal(self.model.hadamard_matrix[target_label], np.squeeze(self.model.predict(clean_img, output_type='codeword'))) # can be deleted after one test


        t = target_label
        # gtl = lambda t,l: np.matmul(logit_P, self.model.hadamard_matrix[t])-np.matmul(logit_P, self.model.hadamard_matrix[l])
        # store all possible gtl first so the whole mat is a constant regardless of t and l, note gtt is 0
        gtls = [np.matmul(logit_P, self.model.hadamard_matrix[t]) - np.matmul(logit_P, self.model.hadamard_matrix[l]) for l in range(n)]
        gtls_no_t = gtls.copy()
        del gtls_no_t[t]
        gols = [np.matmul(logit_P, self.model.hadamard_matrix[clean_label]) - np.matmul(logit_P, self.model.hadamard_matrix[l]) for l in range(n)]
        gols_no_o = gols.copy()
        del gols_no_o[clean_label]


        # conf = np.max(np.abs(gtls_no_t)/n)
        conf = np.min(np.abs(gols_no_o) / n)
        # formal def of inner min part of eq.(17) that should be minimized
        a_func = lambda at: at**2 + np.sum( [ min(0, at + g / n -conf)**2 for g in gtls_no_t] )
        # a_func = lambda at: at ** 2 + np.sum([min(conf, at  + g / n  ) ** 2 for g in gols_no_o])
        # find optimized a_star
        a_t = optimize.brent(a_func)

        # opt_a = np.zeros(n)

        # opt_a = [min(0, a_t + gtls[i] / n - conf) for i in range(n)]
        opt_a = [min( 0, a_t -conf + gtls[i] / n  ) for i in range(n)]
        opt_a[t] = a_t


        d_star = np.matmul(opt_a, self.model.hadamard_matrix.T)

        Jacobian_J = self._Jacbian_logits_fn([clean_img])[0]
        J_matrix = Jacobian_J.reshape(32, np.prod(clean_img.shape))
        if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
            # no advs can be found, attack is invalid
            is_adv, linf, psnr = ecoc_NewJacob_adv.new_perturbed(clean_img, output_type='codewords', num_iter=iter)
            ecoc_NewJacob_adv.update_records(stop='0_Jacb')
            return ecoc_NewJacob_adv
        # check the rank of J_matrix
        J_rank = np.linalg.matrix_rank(J_matrix)

        J = np.matrix(J_matrix)
        try:
            perturbetion = J.T*(J*J.T).I*d_star.reshape(n,1)
            # perturbetion = J.I*d_star.reshape(n,1)
            # perturbetion = 0.03*perturbetion / np.sqrt(np.sum(np.square(perturbetion)))
            # perturbetion = np.clip(perturbetion, -0.5, 0.5)
            # perturbed = np.array(perturbetion).reshape(clean_img.shape)
        except:
            perturbetion = J_matrix.T * np.linalg.pinv(J_matrix * J_matrix.T) * d_star.reshape(n, 1)
            # perturbetion = np.linalg.pinv(J_matrix) * d_star.reshape(n, 1)
            # perturbetion = np.clip(perturbetion, -0.5,0.5)
            # perturbetion = 0.03*perturbetion / np.sqrt(np.sum(np.square(perturbetion)))
            # perturbed = np.array(perturbetion).reshape(clean_img.shape)
        a = np.squeeze(np.matmul(J_matrix, perturbetion))
        mse_d_Jper = np.mean(np.square(d_star - a))

        percent_p_big = np.array([np.sum(np.abs(perturbetion) > x) / np.prod(np.array(perturbetion).shape) for x in [1, 0.8, 0.5]])
        # perturbetion = np.clip(perturbetion, -0.5, 0.5)

        perturbed = np.clip(clean_img + np.array(perturbetion).reshape(clean_img.shape), 0, 1)
        perturbation_psnr = compute_psnr(clean_img, clean_img + perturbed)

        ecoc_NewJacob_adv.new_perturbed(perturbed,output_type = 'codewords')
        ecoc_NewJacob_adv.update_records(J_rank = J_rank, perturbetion_psnr = perturbation_psnr, percent_perturbetion_out_range_elements = percent_p_big,
                                         mse_d_Jper = mse_d_Jper, confidence=conf)
        if ecoc_NewJacob_adv.best_adv is None:
            clean_output = logit_P
            perturbed_output = self.model.predict(perturbed, 'activated')
            perturbed_codeword = self.model.predict(perturbed, 'codeword')
            perturbed_label = self.model.predict(perturbed, 'hard_label')
            ecoc_NewJacob_adv.update_records(clean_logits=clean_output, perturbed_logits = perturbed_output,
                                             perturbed_codeword = perturbed_codeword, perturbed_label = perturbed_label)

        return ecoc_NewJacob_adv

    def OnestepJacAttackv2(self, clean_img, target_codeword, **kwargs):
        """
        anothor temp version
        """
        def Pq_solver(P, q, G=None, h=None, A=None, b=None):
            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P)
            q = matrix(q)
            if G is not None:
                G = matrix(G)
            if h is not None:
                h = matrix(h)
            if A is not None:
                A = matrix(A)
            if b is not None:
                b = matrix(b)
            sol = solvers.qp(P, q, G, h, A, b)

            return sol

        ecoc_NewJacob_adv = UltimateAdversarial(clean_img, is_target_attack=True, target_output=target_codeword,
                                               latent_model=self.model, better_criterion='psnr', output_type='codewords',
                                           conf_thres = None, strict_conf_thres = False) # really careful
        try:
            logit_type = kwargs['logit_type']
            if logit_type == 'activated':
                logit_P = self.model.predict(clean_img, output_type='activated')  # v2.0
            else:
                logit_P = self.model.predict(clean_img,
                                             output_type='logits')  # network output after sigmoid before others
        except KeyError:
            logit_P = self.model.predict(clean_img, output_type='logits')  # network output after sigmoid before others
            pass

        try:
            confidence = kwargs['confidencce']
        except KeyError:
            confidence = 0

        # logit_P = self.model.predict(clean_img, output_type='activated')  # v2.0
        n = len(self.model.hadamard_matrix[0])

        target_label = np.argmax(np.matmul(target_codeword, self.model.hadamard_matrix.T)) # denoted as t in the report
        clean_label = np.argmax(np.matmul(logit_P, self.model.hadamard_matrix))
        Jacobian_J = self._Jacbian_logits_fn([clean_img])[0]
        J_matrix = Jacobian_J.reshape(32, np.prod(clean_img.shape))

        _n = list(range(n))
        _n.remove(target_label)
        A = np.array([self.model.hadamard_matrix[i] - self.model.hadamard_matrix[target_label] for i in _n])
        if A.shape != (n-1,n):
            A = A.transpose()
            assert A.shape == (n-1, n)
        b = gtls = np.array([np.matmul(logit_P, self.model.hadamard_matrix[target_label]) - np.matmul(logit_P, self.model.hadamard_matrix[l]) for l in _n])
        # if apply confidence
        b = b - confidence * np.ones_like(b)
        solution = Pq_solver(P = np.matmul(np.matmul(A,J_matrix),np.matmul(J_matrix.transpose(),A.transpose())), q = b, G = -np.eye(n-1), h = np.zeros(n-1))
        lambda_star = np.array(solution['x'])
        d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
        perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))


        if np.sum(np.square(J_matrix)) == 0:  # somehow Jacobian is 0
            # no advs can be found, attack is invalid
            is_adv, linf, psnr = ecoc_NewJacob_adv.new_perturbed(clean_img, output_type='codewords', num_iter=iter)
            ecoc_NewJacob_adv.update_records(stop='0_Jacb')
            return ecoc_NewJacob_adv
        # check the rank of J_matrix
        J_rank = np.linalg.matrix_rank(J_matrix)


        a = np.squeeze(np.matmul(J_matrix, perturbetion_x_star))
        mse_d_Jper = np.mean(np.square(d_star - a))

        percent_p_big = np.array([np.sum(np.abs(perturbetion_x_star) > x) / np.prod(np.array(perturbetion_x_star).shape) for x in [1, 0.75, 0.5]])
        # perturbetion = np.clip(perturbetion, -0.5, 0.5)

        # perturbed = np.clip(clean_img + np.array(perturbetion_x_star).reshape(clean_img.shape), 0, 1)
        perturbed = clean_img + np.array(perturbetion_x_star).reshape(clean_img.shape)
        perturbation_psnr = compute_psnr(clean_img, perturbed)


        ecoc_NewJacob_adv.new_perturbed(perturbed, output_type = 'codewords')
        ecoc_NewJacob_adv.update_records(J_rank = J_rank, perturbetion_psnr = perturbation_psnr, percent_perturbetion_out_range_elements = percent_p_big,
                                         mse_d_Jper = mse_d_Jper, confidence=confidence)
        if ecoc_NewJacob_adv.best_adv is None:
            clean_output = logit_P
            perturbed_output = self.model.predict(perturbed, 'activated')
            perturbed_codeword = self.model.predict(perturbed, 'codeword')
            perturbed_label = self.model.predict(perturbed, 'hard_label')
            ecoc_NewJacob_adv.update_records(clean_logits=clean_output, perturbed_logits = perturbed_output,
                                             perturbed_codeword = perturbed_codeword, perturbed_label = perturbed_label)

        return ecoc_NewJacob_adv

    def IterativeJacobAttack_NewJac_v2(self, clean_img, targeting_codeword, max_iteration,  **kwargs):
        """
        anthor temp version, very close to the final version
        """

        def Pq_solver(P, q, G=None, h=None, A=None, b=None):
            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P)
            q = matrix(q)
            if G is not None:
                G = matrix(G)
            if h is not None:
                h = matrix(h)
            if A is not None:
                A = matrix(A)
            if b is not None:
                b = matrix(b)
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            return sol

        def new_Jac_perturbetion_v2(target_codeword, logit_P, J_matrix, confidence):
            '''copy fromm new attak'''
            k = len(self.model.hadamard_matrix)
            n = len(self.model.hadamard_matrix[0])

            target_label = np.argmax(
                np.matmul(target_codeword, self.model.hadamard_matrix.T))  # denoted as t in the report
            clean_label = np.argmax(np.matmul(logit_P, self.model.hadamard_matrix.T))

            _n = list(range(k))
            _n.remove(target_label)
            A = np.array([self.model.hadamard_matrix[i] - self.model.hadamard_matrix[target_label] for i in _n])
            if A.shape != (k - 1, n):
                A = A.transpose()
                assert A.shape == (k - 1, n), 'A.shape={},n={}'.format(A.shape,n)
            b = gtls = np.array([np.matmul(logit_P, self.model.hadamard_matrix[target_label]) - np.matmul(logit_P,
                                                                                                          self.model.hadamard_matrix[
                                                                                                              l]) for l
                                 in _n], dtype=np.float)
            # if apply confidence
            b = b - confidence * np.ones_like(b)
            solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose()), dtype=np.float),
                                 q=b, G=-np.eye(k - 1), h=np.zeros(k - 1))
            lambda_star = np.array(solution['x'])
            d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
            perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
            return perturbetion_x_star, d_star, lambda_star

        try:
            logit_type = kwargs['logit_type']
            assert logit_type in ['logit', 'activated']
        except KeyError:
            logit_type = 'logit'

        try:
            confidence = kwargs['confidence']
        except KeyError:
            confidence = 0

        try:
            normalize_direction = kwargs['normalize_direction']
        except KeyError:
            normalize_direction = False
        if normalize_direction == True:
            try:
                step_size = kwargs['step_size']
            except KeyError:
                print('no step size provided, no normalization applied')
                normalize_direction = False

        ecoc_Jacob_adv = UltimateAdversarial(clean_img, is_target_attack=True, target_output=targeting_codeword,
                                             latent_model=self.model, better_criterion='psnr', output_type='codewords',
                                             conf_thres = None, strict_conf_thres = False) # really careful
        clean_codeword = ecoc_Jacob_adv.clean_output

        perturbed = clean_img
        J_ranks = []
        perturbetion_records = []
        # updating iteratively
        for iter in range(max_iteration):
            # current_logits = self.model.predict(perturbed, output_type='logits') # v1.1
            current_logits = self.model.predict(perturbed, output_type=logit_type) # v1.0
            # if iter%50 == 0:
            #     print(current_logits)

            # Jacobian_J = self._Jacbian_fn([perturbed])[0] # v1.0: derivative at activated level

            Jacobian_J = self._Jacbian_logits_fn([perturbed])[0] # v1.1: derivative at logits level

            J_matrix = Jacobian_J.reshape(len(targeting_codeword), np.prod(clean_img.shape))
            # print(Jacobian_J.shape)
            if np.sum(np.square(Jacobian_J)) == 0: #somehow Jacobian is 0
                # no advs can be found, attack is invalid
                is_adv, linf, psnr = ecoc_Jacob_adv.new_perturbed(perturbed, output_type='codewords', num_iter=iter)
                ecoc_Jacob_adv.update_records(zero_J_stop_at=iter)
                break
            # check the rank of J_matrix
            J_rank = np.linalg.matrix_rank(J_matrix)
            J_ranks.append(J_rank)
            # if J_rank == 32 or J_rank==np.prod(clean_img.shape):
            #     ecoc_Jacob_adv.update_records(full_rank = True)
            perturbetion, d_star, lambda_star = new_Jac_perturbetion_v2(targeting_codeword, current_logits, J_matrix, confidence)
            perturbetion_records.append(perturbetion)
            if normalize_direction:
                perturbetion = step_size * perturbetion / np.sqrt(np.sum(np.square(perturbetion)))
            perturbed = perturbed + np.array(perturbetion).reshape(clean_img.shape)

            is_adv, linf, psnr = ecoc_Jacob_adv.new_perturbed(perturbed, output_type = 'codewords', num_iter=iter)

        ecoc_Jacob_adv.update_records(perturbetion_records = np.array(perturbetion_records))


        max_iter_psnr = compute_psnr(perturbed, clean_img)
        ecoc_Jacob_adv.update_records(J_ranks=tuple(J_ranks), max_iter_psnr = max_iter_psnr)

        return  ecoc_Jacob_adv

class ECOCAttack_multlabel(object):
    def __init__(self, model):
        self.model = model

        # get logits before activation (softmax)
        logits = model.output.op.inputs[0]
        logits_list = tf.unstack(logits, axis=1)
        logits_list = [tf.reshape(li, [-1, 1]) for li in logits_list]
        self.N = model.output_shape[1]
        # logits = [activate.op.inputs[0] for activate in individual_activations]
        self.output_codebook = np.eye(self.N)
        self._logits_func = K.function([model.input], [logits])

        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, model.input) for li in logits_list]
        Jacbs_logits = tf.stack(Jacbs_logits_list, name='Jacb_logits')
        self._Jacbian_logits_fn = K.function([model.input],[Jacbs_logits])

        # can be used to valid gradient vanished at activation level
        logits_grad = K.gradients(logits, model.input)
        self._logits_grad_fn = K.function([model.input], logits_grad)
        dodl = K.gradients(model.output, logits)
        self._dodl_grad_fn = K.function([model.input], dodl)

        # the bit models are trained using hinge losses function
        # y_trues serves as both ground truth and attacking target label
        y_trues = K.placeholder(shape=model.output_shape[1])
        # hinge_losses = [keras.losses.hinge(y_pred=y_pred, y_true=y_trues[i]) for i, y_pred in enumerate(logits_list)]
        # self._hinge_losses_func = K.function([model.input, y_trues], hinge_losses)

        # confidence loss
        confidence_thres = K.placeholder(shape=(None, 1))
        # thres been expand with a new axis to be aglined with logits, do this when call gradient func
        confidence_losses = K.min([z * 2 * y_trues[i] for i, z in enumerate(logits_list)] + [confidence_thres])
        self._confidence_losses_func = K.function([model.input, y_trues, confidence_thres], [confidence_losses])
        dcdx = K.gradients(confidence_losses, model.input)  # derivative of confidence over image input x
        self._confidence_gradient_func = K.function([model.input, y_trues, confidence_thres], dcdx)


    def CWBattack(self, clean_img, confidence=0.0, initial_constant=0.01, learning_rate=5e-3,
                  num_binary_search=10, max_iterations=1000, targeting_un_decoded=None):

        """
        This is the BowenAttack that we proposed in the ECOC paper

        :param clean_img:
        :param confidence:
        :param initial_constant:
        :param learning_rate:
        :param num_binary_search:
        :param max_iterations:
        :param targeting_un_decoded: {-1, +1} obtained directly by sign function
        :return:
        """
        ecoc_CWB_adv = UltimateAdversarial_multilabel(clean_img, is_target_attack=True, target_output=targeting_un_decoded,
                                           latent_model=self.model, better_criterion='psnr',
                                           conf_thres=confidence, strict_conf_thres=True)  # really careful
        # variables will be used for whole attack
        # clean_code = ecoc_CWB_adv.clean_output
        min_, max_ = (0, 1)
        # target_error = ecoc_CWB_adv.target_error
        # adam algorithm for gradient updating
        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_img.shape)

        # target loss should be aligned with target code, i.e. using 1 indicates those unchanged label and -1 those changed
        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999  # 0.9999999999999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(w):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            w = np.tanh(w)

            grad = 1 - np.square(w)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            w = w * b + a

            grad = grad * b
            return w, grad

        def overall_loss_grad(img, reconstruct_clean_img, raw_tar_code, constant, confidence):
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty

            confidence = np.array(confidence)
            while confidence.ndim < 2:
                confidence = confidence[np.newaxis]

            confidence_loss = self._confidence_losses_func([img, raw_tar_code, confidence])[
                0]  # this loss should be maxmized to get desired error
            # print('confidence loss:', confidence_loss)

            confidence_grad = self._confidence_gradient_func([img, raw_tar_code, confidence])[0]

            # L2 distance loss, grad w.r.t. img
            squared_l2_distance = np.sum((img - reconstruct_clean_img) ** 2) / s ** 2
            squared_l2_distance_grad = (2 / s ** 2) * (img - reconstruct_clean_img)

            # total loss, grad w.r.t. img
            overall_loss = squared_l2_distance - constant * confidence_loss
            overall_grad = squared_l2_distance_grad - constant * confidence_grad
            return overall_loss, overall_grad

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_clean_img = to_attack_space(clean_img)
        # will be close but not identical to clean_img
        reconstructed_clean_img, _ = to_model_space(att_clean_img)

        # binary search for good constant c
        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_constant
        lower_bound = 0
        upper_bound = np.inf
        for binary_search_step in range(num_binary_search):
            att_perturbation = np.zeros_like(att_clean_img)
            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf
            for iteration in range(max_iterations):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)
                # compute current confidence
                current_logits = np.squeeze(np.array(self._logits_func([perturbed])))
                bit_confidences = 2 * targeting_un_decoded * current_logits
                current_confidence = np.min(bit_confidences)

                is_adv, linf, psnr = ecoc_CWB_adv.new_perturbed(perturbed=perturbed, constant=const,
                                                                num_iteration=iteration, conf=current_confidence,
                                                                num_binary_search=binary_search_step)
                # is_adv, linf, per_predictions, per_predict_codeword = self._is_adv(reconstructed_clean_img, perturbed,
                #                                                                    target_code)
                model_space_loss, model_space_grad = overall_loss_grad(perturbed, reconstructed_clean_img,
                                                                       targeting_un_decoded,
                                                                       const, confidence)
                # print('total_loss in model space = %.f', model_space_loss)
                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grad * dpertub_dw
                gradient_vanished = True if np.sum(np.square(att_space_grad)) == 0 else False
                if gradient_vanished == True:
                    ecoc_CWB_adv.update_records(gradient_vanished=True)
                    assert np.sum(np.square(optimizer(att_space_grad, learning_rate))) == 0
                    break
                att_perturbation -= optimizer(att_space_grad, learning_rate)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True
                    # break # for saving time

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    # _activated = self.model.predict(perturbed, output_type='activated')
                    # print(_activated)
                    # print('is adv = {}, adversarial.idv_found = {}'
                    #       .format(np.array_equal(np.sign(_activated), targeting_un_decoded),
                    #               ecoc_CWB_adv.best_adv is not None)                          )
                    # print('model space loss = {}'.format(model_space_loss))
                    # after each tenth of the iterations, check progress
                    if not (model_space_loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = model_space_loss

            # binary search part
            if found_adv:
                # logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                # logging.info(
                #     "failed to find adversarial " "with const = {}".format(const)
                # )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

        return ecoc_CWB_adv

class KerasModelCW(object):
    """
    This class if for CW attack against common one-hot models, the __init__ method is for construct useful keras functions
    CW_attack method is the real CW attack(my implementation). As this class share the same class structure as the ECOCAttack
    class (python class for initialization, class method for running attack), this class is only used in the experiments of
    ECOC paper
    """

    def __init__(self, wrapped_model):
        self.model = wrapped_model
        keras_model = wrapped_model.keras_model

        # get logits before activation (tanh)
        logits = keras_model.output.op.inputs[0]
        # logits = [activate.op.inputs[0] for activate in individual_activations]
        self._logits_func = K.function([keras_model.input], logits)
        # can be used to valid gradient vanished at activation level
        logits_grad = K.gradients(logits, keras_model.input)
        self._logits_grad_fn = K.function([keras_model.input], logits_grad)
        dodl = K.gradients(keras_model.output, logits)
        self._dodl_grad_fn = K.function([keras_model.input], dodl)
        # the bit models are trained using hinge losses function
        # y_trues serves as both ground truth and attacking target label
        y_trues = K.placeholder(shape=keras_model.output_shape[1])
        # hinge_losses = [keras.losses.hinge(y_pred=y_pred, y_true=y_trues[i]) for i, y_pred in
        #                 enumerate(individual_activations)]
        # self._hinge_losses_func = K.function([keras_model.input, y_trues], hinge_losses)

        # confidence loss
        confidence_thres = K.placeholder(shape=(None, 1))
        # thres been expand with a new axis to be aglined with logits, do this when call gradient func
        # confidence_losses = K.min([z * 2 * y_trues[i] for i, z in enumerate(logits)] + [confidence_thres])
        # self._confidence_losses_func = K.function([keras_model.input, y_trues, confidence_thres], [confidence_losses])
        # dcdx = K.gradients(confidence_losses, keras_model.input)  # derivative of confidence over image input x
        # self._confidence_gradient_func = K.function([keras_model.input, y_trues, confidence_thres], dcdx)

        # eoco logits
        # hadamard = K.variable(self.model.hadamard_matrix.T)
        # ecoc_logits = K.dot(keras_model.output, hadamard) # shape: (num_img, 32) * (32, 32), hadamard matrix is Symmetric
        # self._ecoc_logits_func = K.function([keras_model.input], [ecoc_logits])
        class_maximize = K.placeholder(dtype='int32')
        class_minimize = K.placeholder(dtype='int32')
        confidence_losses = K.max([logits[:,class_minimize] - logits[:,class_maximize], -confidence_thres[:,0]])
        self._confidence_loss_func = K.function([keras_model.input, class_minimize, class_maximize, confidence_thres], [confidence_losses])
        confidence_grad = K.gradients(confidence_losses, keras_model.input)
        self._confidence_grad_fn = K.function([keras_model.input, class_minimize, class_maximize, confidence_thres], confidence_grad)

    def CW_Attack(self, clean_img, confidence=0.0, initial_constant=0.01, learning_rate=5e-3,
                       num_binary_search=10, max_iterations=1000, target_class=None):

        """

        :param clean_img:
        :param confidence:
        :param initial_constant:
        :param learning_rate:
        :param num_binary_search:
        :param max_iterations:
        :param targeting_ecoc_logits_num: targeting "class" by sorting the ecoc logits, same as ground label is non-target attack
        :return:
        """
        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(w):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            w = np.tanh(w)

            grad = 1 - np.square(w)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            w = w * b + a

            grad = grad * b
            return w, grad

        def overall_loss_grad(img, reconstruct_clean_img, class_minimize, class_maximize, constant, confidence):
            """

            :param img:
            :param reconstruct_clean_img:
            :param who_minus_whom: # e.g. [ 0, 0, 1, -1, 0, 0]  indicates the third minus the forth
            :param constant:
            :param confidence:
            :return:
            """
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty
            # gradient_vanished = False
            confidence = np.array(confidence)
            while confidence.ndim < 2:
                confidence = confidence[np.newaxis]

            confidence_loss = self._confidence_loss_func([img, class_minimize, class_maximize, confidence])[
                0]  # this loss should be maxmized to get desired error

            confidence_grad = self._confidence_grad_fn([img, class_minimize, class_maximize, confidence])[0]

            # L2 distance loss, grad w.r.t. img
            squared_l2_distance = np.sum((img - reconstruct_clean_img) ** 2) / s ** 2
            squared_l2_distance_grad = (2 / s ** 2) * (img - reconstruct_clean_img)

            # total loss, grad w.r.t. img
            overall_loss = squared_l2_distance + constant * confidence_loss
            overall_grad = squared_l2_distance_grad + constant * confidence_grad
            return overall_loss, overall_grad

        # def compute_confidence(img, class_minimize, class_maximize):
        #
        #     return  ecoc_logits[:,class_minimize] - ecoc_logits[:,class_maximize]

        is_targeted = False if target_class == None else True
        print('the target output input to Adversarial object is ', target_class)
        keras_CW_adv = UltimateAdversarial(clean_img, is_target_attack=is_targeted, target_output=target_class,
                                        latent_model=self.model, better_criterion='psnr', output_type='hard_label',
                                          conf_thres = confidence, strict_conf_thres = True)

        # variables will be used for whole attack
        clean_output = keras_CW_adv.clean_output
        min_, max_ = (0, 1)
        target_error = keras_CW_adv.target_error
        # adam algorithm for gradient updating
        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_img.shape)

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_clean_img = to_attack_space(clean_img)
        # will be close but not identical to clean_img
        reconstructed_clean_img, _ = to_model_space(att_clean_img)

        # binary search for good constant c
        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_constant
        lower_bound = 0
        upper_bound = np.inf
        for binary_search_step in range(num_binary_search):
            att_perturbation = np.zeros_like(att_clean_img)
            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf
            for iteration in range(max_iterations):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)

                # do one forward to get ecoc logits and find how who should minus whom
                logits = np.squeeze(self._logits_func([perturbed])[0])
                if keras_CW_adv.target_output is None:
                    # print('========HUGE WARNING=======\nThis supposed to be a target-attack script (for now)')
                    class_minimize = int(keras_CW_adv.clean_output)
                    x = np.zeros_like(logits)
                    x[class_minimize] = np.inf
                    class_maximize = np.argmax(logits - x)
                else:
                    class_maximize = target_class
                    x = np.zeros_like(logits)
                    x[target_class] = np.inf
                    class_minimize = np.argmax(logits-x)

                current_confidence = logits[class_maximize] - logits[class_minimize]

                is_adv, linf, psnr = keras_CW_adv.new_perturbed(perturbed=perturbed, constant=const,
                                                                num_iteration=iteration, conf=current_confidence)

                model_space_loss, model_space_grad = overall_loss_grad(perturbed, reconstructed_clean_img,
                                                                       class_minimize, class_maximize,
                                                                       const, confidence)
                # ecoc_logits[class_minimize] - ecoc_logits[class_maximize] == model_space_loss should be true

                # print('total_loss in model space = %.f', model_space_loss)
                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grad * dpertub_dw

                gradient_vanished = True if np.sum(att_space_grad) == 0 else False

                if gradient_vanished == True:
                    keras_CW_adv.update_records(gradient_vanished=True)
                    assert np.sum(optimizer(att_space_grad, learning_rate)) == 0
                    break
                att_perturbation -= optimizer(att_space_grad, learning_rate)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    # print('activated output', self.model.predict(perturbed, output_type='activated'))
                    # print('ECOC logits',ecoc_logits)
                    # print('model space loss', model_space_loss)

                    # after each tenth of the iterations, check progress
                    if not (model_space_loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = model_space_loss

            # binary search part
            if found_adv:
                # logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                # logging.info(
                #     "failed to find adversarial " "with const = {}".format(const)
                # )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

        return keras_CW_adv

"""
Start from here, the codes below is the New one for experiments of the Jacobian attack

different to the last ECOCAttack class, from here each attack is build as one python class, and initialized independently
"""

class ECOCbase(object):
    def __init__(self, ecoc_model):
        if hasattr(ecoc_model, 'keras_model'):
            self.model = ecoc_model
            # keras model is just for backend opreation, while LatentModel instance is for prediction and other stuff
            assert isinstance(ecoc_model.keras_model, Model), 'ecoc_model.keras_model:{},\n' \
                                                              'Model class:{}'.format(type(ecoc_model.keras_model), type(Model))
        else:
            raise TypeError('Input model must be ethier customed wrapped model')


class JacbMDisTryAndOptimize(ECOCbase):
    """
    the final version of Jacobian attack against ECOC model
    """
    def __init__(self, ecoc_model):
        super(JacbMDisTryAndOptimize, self).__init__(ecoc_model)
        # get logits before activation (tanh)
        keras_model = ecoc_model.keras_model
        individual_activations = keras_model.layers[-1].input
        logits_list = [activate.op.inputs[0] for activate in individual_activations]
        logits_tensor = K.concatenate(logits_list)
        self._logits_func = K.function([keras_model.input], [logits_tensor])

        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, keras_model.input) for li in logits_list]
        Jacbs_logits = tf.stack(Jacbs_logits_list, name='Jacb_logits')
        self._Jacbian_logits_fn = K.function([keras_model.input],[Jacbs_logits])

    def new_perturbed(self, perturbeds,  targeting_codeword, select_best_dis = False, current_logtis = None):
        if current_logtis is None:
            current_logtis = self._logits_func([perturbeds])[0]
        current_labels = np.argmax(np.matmul(current_logtis, self.model.hadamard_matrix.T), axis=-1)
        current_codewords = self.model.hadamard_matrix[current_labels]
        is_adv = np.prod(current_codewords == targeting_codeword, axis=-1)
        psnr = compute_psnr(perturbeds, self.clean_img)
        if select_best_dis:
            if psnr > self.best_psnr:
                self.best_adv = perturbeds
                self.best_psnr = psnr
        return is_adv, psnr

    def run_attack(self, clean_img, targeting_codeword,  **kwargs):

        def Pq_solver(P, q, G=None, h=None, A=None, b=None):
            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P.astype(np.double))
            q = matrix(q.astype(np.double))
            if G is not None:
                G = matrix(G.astype(np.double))
            if h is not None:
                h = matrix(h.astype(np.double))
            if A is not None:
                A = matrix(A.astype(np.double))
            if b is not None:
                b = matrix(b.astype(np.double))
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            return sol

        def generate_perturbetion(target_codeword, logit_P, J_matrix, confidence):
            '''copy fromm new attak'''
            k = len(self.model.hadamard_matrix)
            n = len(self.model.hadamard_matrix[0])

            target_label = np.argmax(
                np.matmul(target_codeword, self.model.hadamard_matrix.T))  # denoted as t in the report
            clean_label = np.argmax(np.matmul(logit_P, self.model.hadamard_matrix.T))

            _n = list(range(k))
            _n.remove(target_label)
            A = np.array([self.model.hadamard_matrix[i] - self.model.hadamard_matrix[target_label] for i in _n])
            if A.shape != (k - 1, n):
                A = A.transpose()
                assert A.shape == (k - 1, n), 'A.shape={},n={}'.format(A.shape, n)
            b = gtls = np.array([np.matmul(logit_P, self.model.hadamard_matrix[target_label]) - np.matmul(logit_P,
                                                                                                          self.model.hadamard_matrix[
                                                                                                              l]) for l
                                 in _n], dtype=np.float)
            # if apply confidence
            b = b - confidence * np.ones_like(b)
            solution = Pq_solver(
                P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose()), dtype=np.float32),
                q=b, G=-np.eye(k - 1), h=np.zeros(k - 1))
            lambda_star = np.array(solution['x'])
            d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
            perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
            return perturbetion_x_star, d_star, lambda_star

        # inis and records
        self.clean_img = clean_img
        self.target_codeword = targeting_codeword
        self.best_adv = None
        self.best_psnr = -np.inf


        try:
            logit_type = kwargs['logit_type']
            assert logit_type in ['logit', 'activated']
        except KeyError:
            logit_type = 'logit'

        try:
            confidence = kwargs['confidence']
        except KeyError:
            confidence = 0

        step_size = kwargs['step_size']
        max_iteration = kwargs['max_iteration']

        perturbed = clean_img
        J_ranks = []
        psnr = 0
        for iteration in range(max_iteration):

            # caculate the first perturbetion and check
            current_logits = self._logits_func([perturbed])[0]
            Jacobian_J = self._Jacbian_logits_fn([perturbed])[0]  # v1.1: derivative at logits level
            J_matrix = Jacobian_J.reshape(len(targeting_codeword), np.prod(clean_img.shape))

            # if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
            #     # no advs can be found, attack is invalid
            #     is_adv, psnr = new_perturbed(perturbed,  targeting_codeword = targeting_codeword,)
            #     return is_adv, perturbed, psnr
            # check the rank of J_matrix
            J_rank = np.linalg.matrix_rank(J_matrix)
            J_ranks.append(J_rank)

            perturbetion, d_star, lambda_star = generate_perturbetion(targeting_codeword, current_logits, J_matrix,
                                                                        confidence)
            perturbetion = perturbetion.reshape(perturbed.shape)
            is_adv, psnr = self.new_perturbed(perturbed + perturbetion, targeting_codeword)

            p_norm = np.sqrt(np.sum(np.square(perturbetion)))
            p_dir = perturbetion / p_norm

            if is_adv:
                # bs search and return
                upper = 1
                under = 0
                while upper - under > 0.01:
                    # number of bs searches
                    mid = (upper + under) / 2
                    perturbetion = mid * p_dir * p_norm
                    is_adv, psnr = self.new_perturbed(perturbed + perturbetion, targeting_codeword, select_best_dis=True)
                    if is_adv:
                        # lower the upper
                        upper = mid
                    else:
                        under = mid
                return self.best_adv, self.best_psnr, iteration
            else:
                # move the perturbed a bit and test again
                perturbed = perturbed + step_size * p_dir
        # all fails

        return None, psnr, max_iteration


class KerasModelAttack(object):
    def __init__(self, model):
        self.model = model


class JacobianLogitsOnehotv2(KerasModelAttack):
    """
    this is one temp version of Jacobian attack on one-hot model, it is almost the same with the final version with the
    difference of records more internal statics,  this class is mainly used for debugging.
    """
    def __init__(self, model):
        super(JacobianLogitsOnehotv2, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax)
        logits = keras_model.output.op.inputs[0]
        logits_list = tf.unstack(logits, axis=1)
        # logits = [activate.op.inputs[0] for activate in individual_activations]
        self.output_codebook = np.eye(keras_model.output_shape[1])
        self._logits_func = K.function([keras_model.input], [logits])
        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, keras_model.input) for li in logits_list]
        Jacbs_logits = tf.stack(values=Jacbs_logits_list,  axis=2, name='JacobLogits')[0]
        self._Jacbian_logits_fn = K.function([keras_model.input], [Jacbs_logits])
        # can be used to valid gradient vanished at activation level
        logits_grad = K.gradients(logits, keras_model.input)
        self._logits_grad_fn = K.function([keras_model.input], logits_grad)
        dodl = K.gradients(keras_model.output, logits)
        self._dodl_grad_fn = K.function([keras_model.input], dodl)

    def run_attack(self, clean_img, targeting_output, max_iteration, **kwargs):
        # ini record
        self.clean_image = clean_img
        self.target_output = targeting_output
        self.best_psnr = 0
        self.best_adv = None
        self.best_iter = 0
        self.best_logits = None

        def Pq_solver(P, q, G=None, h=None, A=None, b=None):
            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P)
            q = matrix(q)
            if G is not None:
                G = matrix(G)
            if h is not None:
                h = matrix(h)
            if A is not None:
                A = matrix(A)
            if b is not None:
                b = matrix(b)
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            return sol

        def generate_perturbetion(target_output, logit_P, J_matrix, confidence):
            '''copy fromm new attak'''
            n = len(self.output_codebook)

            target_label = np.argmax(
                np.matmul(target_output, self.output_codebook.T))  # denoted as t in the report
            clean_label = np.argmax(np.matmul(logit_P, self.output_codebook))

            _n = list(range(n))
            _n.remove(target_label)
            A = np.array([self.output_codebook[i] - self.output_codebook[target_label] for i in _n])
            if A.shape != (n - 1, n):
                A = A.transpose()
                assert A.shape == (n - 1, n)
            b = gtls = np.array([np.matmul(logit_P, self.output_codebook[target_label])
                                 - np.matmul(logit_P, self.output_codebook[l]) for l in _n])
            # if apply confidence
            b = b - confidence * np.ones_like(b)
            solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
                                 q=b, G=-np.eye(n - 1), h=np.zeros(n - 1))
            lambda_star = np.array(solution['x'])
            d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
            perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
            return perturbetion_x_star, d_star, lambda_star

        def new_perturbed(perturbed, **kwargs):
            try:
                current_logits= kwargs['current_logits']
            except:
                current_logits = self._logits_func([perturbed])

            try:
                current_iter = kwargs['current_iter']
            except:
                current_iter = None

            current_label = np.argmax(current_logits)
            target_label = np.argmax(targeting_output)
            is_adv = np.equal(current_label, target_label)
            psnr = compute_psnr(perturbed, self.clean_image)
            linf = np.max(np.abs(perturbed - self.clean_image))

            if is_adv and psnr > self.best_psnr:
                self.best_adv = perturbed
                self.best_psnr = psnr
                self.best_iter = current_iter


            return is_adv, linf, psnr


        try:
            confidence = kwargs['confidence']
        except KeyError:
            confidence = 0

        try:
            normalize_direction = kwargs['normalize_direction']
            if normalize_direction == True:
                try:
                    step_size = kwargs['step_size']
                except KeyError:
                    print('no step size provided, no normalization applied')
                    assert  normalize_direction == False, 'either no Norm or valid step size'
        except KeyError:
            try:
                step_size = kwargs['step_size']
                normalize_direction = True
            except KeyError:
                print('no step size provided, no normalization applied')
                normalize_direction = False



        perturbed = clean_img
        J_ranks = []
        # updating iteratively
        for iter in range(max_iteration):
            # current_logits = self.model.predict(perturbed, output_type='logits') # v1.1
            current_logits = self._logits_func([perturbed])[0]
            if iter%50 == 0:
                print(current_logits)

            # Jacobian_J = self._Jacbian_fn([perturbed])[0] # v1.0: derivative at activated level

            Jacobian_J = self._Jacbian_logits_fn([perturbed])[0]  # v1.1: derivative at logits level

            J_matrix = Jacobian_J.reshape(32, np.prod(clean_img.shape))
            # print(Jacobian_J.shape)
            if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
                # no advs can be found, attack is invalid
                is_adv, linf, psnr = new_perturbed(perturbed, current_logits=current_logits, current_iter=iter)
                break
            # check the rank of J_matrix
            J_rank = np.linalg.matrix_rank(J_matrix)
            J_ranks.append(J_rank)
            # if J_rank == 32 or J_rank==np.prod(clean_img.shape):
            #     ecoc_Jacob_adv.update_records(full_rank = True)
            perturbetion, d_star, lambda_star = generate_perturbetion(targeting_output, current_logits,
                                                                        J_matrix, confidence)
            if normalize_direction:
                perturbetion = step_size * perturbetion / np.sqrt(np.sum(np.square(perturbetion)))

            perturbed = perturbed + np.array(perturbetion).reshape(clean_img.shape)

            is_adv, linf, psnr = new_perturbed(perturbed, current_logits=current_logits, current_iter=iter)

        max_iter_psnr = compute_psnr(perturbed, clean_img)


        return self.best_adv, self.best_psnr, self.best_iter, max_iter_psnr


class JacobMDisTryAndOptimizeOnehot(KerasModelAttack):
    """
    This is the final Jacobian attack (class) for the one-hot model
    """

    def __init__(self, model):
        super(JacobMDisTryAndOptimizeOnehot, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax)
        logits = keras_model.output.op.inputs[0]
        logits_list = tf.unstack(logits, axis=1)
        # logits = [activate.op.inputs[0] for activate in individual_activations]
        self.output_codebook = np.eye(keras_model.output_shape[1])
        self._logits_func = K.function([keras_model.input], [logits])
        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, keras_model.input) for li in logits_list]
        Jacbs_logits = tf.stack(values=Jacbs_logits_list,  axis=2, name='JacobLogits')[0]
        self._Jacbian_logits_fn = K.function([keras_model.input], [Jacbs_logits])
        # can be used to valid gradient vanished at activation level
        logits_grad = K.gradients(logits, keras_model.input)
        self._logits_grad_fn = K.function([keras_model.input], logits_grad)
        dodl = K.gradients(keras_model.output, logits)
        self._dodl_grad_fn = K.function([keras_model.input], dodl)


    def new_perturbed(self, perturbeds,  target_output, select_best_dis = False, current_logits = None):
        # Aug 24: modifying for VOC Multilable
        # if current_logits is None:
        #     current_logits = self._logits_func([perturbeds])[0]
        # current_label = np.argmax(current_logits)
        # target_label = np.argmax(target_output)
        # is_adv = np.equal(current_label, target_label)

        # given target_ouput is one-hot or multi-hot vector
        final_activation_type = self.model.layers[-1].get_config()['activation'] # strs in common activation functions
        predicts = np.squeeze(self.model.predict(perturbeds))
        if final_activation_type == 'sigmoid':
            current_output = (1 + np.sign(np.squeeze(self.model.predict(perturbeds)) - 0.5)) / 2
        elif final_activation_type == 'softmax':
            current_output = np.argmax(predicts)
            target_class = np.argmax(softmax(target_output))
        else:
            raise NotImplementedError('An activation function other than sigmoid/softmax detected, metric of valid adv shall be specified.')
        is_adv = np.array_equal(current_output, target_class)
        psnr = compute_psnr(perturbeds, self.clean_image)
        if select_best_dis:
            if psnr > self.best_psnr:
                self.best_adv = perturbeds
                self.best_psnr = psnr
        return is_adv, psnr

    def run_attack(self, clean_img, targeting_output, **kwargs):
        # ini record
        self.clean_image = clean_img
        self.target_output = targeting_output
        self.best_psnr = 0
        self.best_adv = None
        self.best_iter = 0
        self.best_logits = None

        def Pq_solver(P, q, G=None, h=None, A=None, b=None):
            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P)
            q = matrix(q)
            if G is not None:
                G = matrix(G)
            if h is not None:
                h = matrix(h)
            if A is not None:
                A = matrix(A)
            if b is not None:
                b = matrix(b)
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            return sol

        def generate_perturbetion(target_output, logit_P, J_matrix, confidence):
            '''copy fromm new attak'''
            n = len(self.output_codebook)

            target_label = np.argmax(
                np.matmul(target_output, self.output_codebook.T))  # denoted as t in the report
            clean_label = np.argmax(np.matmul(logit_P, self.output_codebook))

            _n = list(range(n))
            _n.remove(target_label)
            A = np.array([self.output_codebook[i] - self.output_codebook[target_label] for i in _n])
            if A.shape != (n - 1, n):
                A = A.transpose()
                assert A.shape == (n - 1, n)
            b = gtls = np.array([np.matmul(logit_P, self.output_codebook[target_label])
                                 - np.matmul(logit_P, self.output_codebook[l]) for l in _n])
            # if apply confidence
            b = b - confidence * np.ones_like(b)
            solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
                                 q=b, G=-np.eye(n - 1), h=np.zeros(n - 1))
            lambda_star = np.array(solution['x'])
            d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
            perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
            return perturbetion_x_star, d_star, lambda_star


        try:
            confidence = kwargs['confidence']
        except KeyError:
            confidence = 0

        step_size = kwargs['step_size']
        max_iteration = kwargs['max_iteration']

        perturbed = clean_img
        J_ranks = []
        for iteration in range(max_iteration):

            # caculate the first perturbetion and check
            current_logits = self._logits_func([perturbed])[0]
            Jacobian_J = self._Jacbian_logits_fn([perturbed])[0]  # v1.1: derivative at logits level
            J_matrix = Jacobian_J.reshape(len(targeting_output), np.prod(clean_img.shape))

            # if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
            #     # no advs can be found, attack is invalid
            #     is_adv, psnr = new_perturbed(perturbed,  targeting_codeword = targeting_codeword,)
            #     return is_adv, perturbed, psnr
            # check the rank of J_matrix
            J_rank = np.linalg.matrix_rank(J_matrix)
            J_ranks.append(J_rank)

            perturbetion, d_star, lambda_star = generate_perturbetion(targeting_output, current_logits, J_matrix,
                                                                        confidence)
            perturbetion = perturbetion.reshape(perturbed.shape)
            is_adv, psnr = self.new_perturbed(perturbed + perturbetion, targeting_output)

            p_norm = np.sqrt(np.sum(np.square(perturbetion)))
            p_dir = perturbetion / p_norm

            if is_adv:
                # bs search and return
                upper = 1
                under = 0
                while upper - under > 0.01:
                    # number of bs searches
                    mid = (upper + under) / 2
                    perturbetion = mid * p_dir * p_norm
                    is_adv, psnr = self.new_perturbed(perturbed + perturbetion, targeting_output, select_best_dis=True)
                    if is_adv:
                        # lower the upper
                        upper = mid
                    else:
                        under = mid
                return self.best_adv, self.best_psnr, iteration, J_ranks
            else:
                # move the perturbed a bit and test again
                perturbed = perturbed + step_size * p_dir
        # all fails
        return None, 0, 0, J_ranks


class JacobMDisTryAndOptimizeMultiLabel(KerasModelAttack):
    """
    This is the Jacobian Attack Specilized for Multi-label model
    """

    def __init__(self, model):
        super(JacobMDisTryAndOptimizeMultiLabel, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax)
        logits = keras_model.output.op.inputs[0]
        logits_list = tf.unstack(logits, axis=1)
        self.N = keras_model.output_shape[1]
        # logits = [activate.op.inputs[0] for activate in individual_activations]
        self.output_codebook = np.eye(self.N)
        self._logits_func = K.function([keras_model.input], [logits])
        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, keras_model.input) for li in logits_list]
        Jacbs_logits = tf.stack(values=Jacbs_logits_list,  axis=2, name='JacobLogits')[0]
        self._Jacbian_logits_fn = K.function([keras_model.input], [Jacbs_logits])
        # can be used to valid gradient vanished at activation level
        logits_grad = K.gradients(logits, keras_model.input)
        self._logits_grad_fn = K.function([keras_model.input], logits_grad)
        dodl = K.gradients(keras_model.output, logits)
        self._dodl_grad_fn = K.function([keras_model.input], dodl)

    def new_perturbed(self, perturbeds,  target_output,  **kwargs):
        # given target_output is one-hot or multi-hot vector
        final_activation_type = self.model.layers[-1].get_config()['activation'] # strs in common activation functions
        predicts = np.squeeze(self.model.predict(perturbeds))
        if final_activation_type == 'sigmoid':
            # map output to {-1,1}
            current_output = np.sign(np.squeeze(self.model.predict(perturbeds)) - 0.5)
        elif final_activation_type == 'softmax':
            current_output = np.argmax(predicts)
        else:
            raise NotImplementedError('An activation function other than sigmoid/softmax detected, metric of valid adv shall be specified.')

        is_adv = np.array_equal(current_output, target_output)
        psnr = compute_psnr(perturbeds, self.clean_image)

        hamming_loss = np.sum(current_output != target_output) / self.N

        if hamming_loss < self.best_hamming_loss:
            # and psnr > self.best_psnr:
            self.best_adv = perturbeds
            self.best_hamming_loss = hamming_loss
            self.best_psnr = psnr
            try: iteration = kwargs['iter']
            except KeyError: iteration = 0
            self.best_iter = iteration

        return is_adv, psnr, hamming_loss

    def Pq_solver(self, P, q, G=None, h=None, A=None, b=None):
        """
        Solves a quadratic program

            minimize    (1/2)*x'*P*x + q'*x
            subject to  G*x <= h
                        A*x = b.


        Input arguments.

            P is a n x n dense or sparse 'd' matrix with the lower triangular
            part of P stored in the lower triangle.  Must be positive
            semidefinite.

            q is an n x 1 dense 'd' matrix.

            G is an m x n dense or sparse 'd' matrix.

            h is an m x 1 dense 'd' matrix.

            A is a p x n dense or sparse 'd' matrix.

            b is a p x 1 dense 'd' matrix or None.

            solver is None or 'mosek'.

            The default values for G, h, A and b are empty matrices with
            zero rows.
        """
        P = matrix(P)
        q = matrix(q)
        if G is not None:
            G = matrix(G)
        if h is not None:
            h = matrix(h)
        if A is not None:
            A = matrix(A)
        if b is not None:
            b = matrix(b)
        solvers.options['show_progress'] = False
        from mosek import iparam
        solvers.options['MOSEK'] = {iparam.log: 0}
        try:
            # sol = solvers.qp(P, q, G, h, A, b)
            sol = solvers.qp(P, q, G, h, A, b, solver='mosek')
            # print('diff solver equal = ', np.array_equal(sol['x'], sol_mosek['x']))
        except ValueError:
            sol = None

        return sol

    def generate_perturbetion(self, target_output, logit_P, J_matrix, confidence):
        # transform target_output to {-1,1}
        target_output = 2 * target_output - 1

        '''copy fromm new attak'''
        logit_P = np.squeeze(logit_P)
        n = len(logit_P)
        A = -np.array(target_output) * np.eye(n)
        b = np.array(target_output) * logit_P
        # if apply confidence
        b = b - confidence * np.ones_like(b)
        solution = self.Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
                             q=b, G=-np.eye(n), h=np.zeros(n))

        lambda_star = np.array(solution['x']) if solution['x'] else np.zeros((20, 1))
        d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
        perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
        return perturbetion_x_star, d_star, lambda_star

    def run_attack(self, clean_img, targeting_output, **kwargs):
        # ini record
        self.clean_image = clean_img
        self.target_output = targeting_output
        self.best_psnr = 0
        self.best_adv = None
        self.best_iter = 0
        self.best_logits = None
        self.best_hamming_loss = 1
        self.best_AE = None

        ''' read kwargs '''
        try: confidence = kwargs['confidence']
        except KeyError: confidence = 0
        # try: metric = kwargs['metric']
        # except KeyError: metric = 'hamming_loss'

        step_size = kwargs['step_size']
        max_iteration = kwargs['max_iteration']
        binary_search_num = kwargs['binary_search_num']
        try: verbos = kwargs['verbos']
        except KeyError: verbos = True
        try:
            debuging_record_file = kwargs['record_file']
        except KeyError:
            debuging_record_file = None

        with open(debuging_record_file, 'a') as f:
            f.write('{}'.format(kwargs))

        before_perturb_img = clean_img
        J_rank_ob = []  # eventually of shape iters * 4
        clean_logits = self._logits_func([clean_img])[0]
        for iteration in range(max_iteration):
            # caculate the first perturbetion and check
            before_perturb_logits = self._logits_func([before_perturb_img])[0]
            Jacobian_J = self._Jacbian_logits_fn([before_perturb_img])[0]  # v1.1: derivative at logits level
            J_matrix = Jacobian_J.reshape(len(targeting_output), np.prod(clean_img.shape))

            # if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
            #     # no advs can be found, attack is invalid
            #     is_adv, psnr = new_perturbed(perturbed,  targeting_codeword = targeting_codeword,)
            #     return is_adv, perturbed, psnr
            # check the rank of J_matrix
            J_rank = int(np.linalg.matrix_rank(J_matrix))
            # J_ranks.append(J_rank)
            # check relationship between low rank and out of bound pixels

            perturbetion, d_star, lambda_star = self.generate_perturbetion(targeting_output, before_perturb_logits, J_matrix,
                                                                        confidence)
            ''' double-checking the generated perturbetion'''
            p_norm = np.sqrt(np.sum(np.square(perturbetion)))
            mse_Jp_dstar = np.mean(np.square(np.matmul(J_matrix, perturbetion) - d_star))
            obs = np.array([np.sum(perturbetion > x)/np.prod(perturbetion.shape) for x in (1, 10, 100)])
            J_rank_ob.append(np.concatenate(([J_rank], obs)))

            perturbetion = perturbetion.reshape(clean_img.shape)

            after_perturb_img = np.clip(before_perturb_img + perturbetion, 0, 1)

            if np.isnan(before_perturb_img).any():
                break
            if np.isnan(after_perturb_img).any():
                break

            is_adv, psnr, hamming_loss = self.new_perturbed(after_perturb_img, targeting_output, iter = iteration)
            after_perturb_logits = self._logits_func([np.clip(before_perturb_img + perturbetion, 0, 1)])[0]

            p_dir = perturbetion / p_norm

            if is_adv:
                # bs search and return
                upper = 1
                under = 0
                # while upper - under > 0.01:
                for i in range(binary_search_num):
                    # number of bs searches
                    mid = (upper + under) / 2
                    perturbetion = mid * p_dir * p_norm
                    after_perturb_img = np.clip(before_perturb_img + perturbetion, 0, 1)
                    is_adv, psnr, hamming_loss = self.new_perturbed(after_perturb_img, targeting_output, select_best_dis=True)
                    if is_adv:
                        # lower the upper
                        upper = mid
                    else:
                        under = mid
                return {'adv': self.best_adv, 'psnr': self.best_psnr, 'hamming_loss': self.best_hamming_loss,
                        'iteration': iteration, 'JacRank': J_rank_ob, 'is_full_success': True}
            else:
                # move the perturbed a bit and test again
                before_perturb_img = np.clip(before_perturb_img + step_size * p_dir, 0, 1)
        # all fails
            if verbos:
                pass
                # print iterative debugging information
                # Jrank, logits changes, targeting output
                # print('\n========== iteration {} =============\n'
                #       'norms and ranks\n'
                #       '        Jacobian matrix rank = {}\n'
                #       '        perturbetion L2: {}\n'
                #       '        mse<J*per, d_star>: {}\n'
                #       '        PSNR = {}\n'
                #       '        outrange percentage = {}'.format(iteration,
                #                                    J_rank,
                #                                    p_norm,
                #                                    mse_Jp_dstar,
                #                                    psnr,
                #                                    obs))
                # print('\nlogits change'
                #       '\n        before perturbetion: {}'
                #       '\n        after perturbetion : {}'
                #       '\n        targeting output:    {}'
                #       '\n        clean logits:        {}'
                #             .format(list(before_perturb_logits.round(4)),
                #                     list(after_perturb_logits.round(4)),
                #                     list(targeting_output.round(0)),
                #                     list(clean_logits.round(4))))
            if debuging_record_file:
                with open(debuging_record_file,  'a') as f:
                    f.write('\n========== iteration {} =============\n'
                      'norms and ranks\n'
                      '        Jacobian matrix rank = {}\n'
                      '        perturbetion L2: {}\n'
                      '        mse<J*per, d_star>: {}\n'
                      '        PSNR = {}\n'
                      '        outrange percentage = {}'.format(iteration,
                                                   J_rank,
                                                   p_norm,
                                                   mse_Jp_dstar,
                                                   psnr,
                                                   obs))
                    f.write('\nlogits change'
                      '\n        before perturbetion: {}'
                      '\n        after perturbetion : {}'
                      '\n        targeting output:    {}'
                      '\n        clean logits:        {}'
                            .format(list(before_perturb_logits.round(4)),
                                    list(after_perturb_logits.round(4)),
                                    list(targeting_output.round(0)),
                                    list(clean_logits.round(4))))
        return {'adv': self.best_adv, 'psnr': self.best_psnr, 'hamming_loss': self.best_hamming_loss,
                         'iteration': self.best_iter, 'JacRank': J_rank_ob, 'is_full_success': False}


class MultiLabelCW(KerasModelAttack):
    """
    This is my implementation of CW attack for multi-label attack, the attack is the same as the multi-label adversarial
    paper( ICDM conference)
    """
    def __init__(self, model):
        super(MultiLabelCW, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax)
        logits = keras_model.output.op.inputs[0]
        # logits_list = tf.unstack(logits, axis=1)
        self._logits_func = K.function([keras_model.input], [logits])

        # ML-CW loss = sum of each bit CW loss
        constant_lambda = K.placeholder(shape=(None, 1))
        target_output = K.placeholder(shape=keras_model.output_shape[1])
        # target_outputs = tf.unstack(target_output)
        confidence = K.placeholder(shape=keras_model.output_shape[1])
        zeros = tf.constant(np.zeros(keras_model.output_shape[1]),dtype=tf.float32)

        CWloss = tf.reduce_sum(tf.maximum(tf.multiply(-target_output, logits) + confidence, zeros))
        # CWloss = constant_lambda * tf.reduce_sum([tf.reduce_max(0, y*l)-conficence for y, l in zip(target_outputs, logits_list)])
        CWloss_grads = K.gradients(CWloss, keras_model.input)
        self._CWloss_fn = K.function([constant_lambda, target_output, keras_model.input, confidence], [CWloss])
        self._CWloss_grads_fn = K.function([constant_lambda, target_output, keras_model.input, confidence], CWloss_grads)

        # TODO: finish all this
    def predict_logits(self, image): return np.squeeze(self._logits_func([image])[0])

    def cw_loss(self, image, constant_lambda, target_output, confidence):
        return self._CWloss_fn([constant_lambda, target_output, image, confidence])[0]

    def cw_loss_grad(self, image, constant_lambda, target_output, confidence):
        return self._CWloss_grads_fn([constant_lambda, target_output, image, confidence])[0]

    def new_perturbed(self, image, target_output, **kwargs):
        # ML model by default has sigmoid activation
        # map prediction to {-1,1}
        const = kwargs['constant']
        current_output = np.sign(np.squeeze(self.model.predict(image)) - 0.5)
        is_adv = np.array_equal(current_output, target_output)
        psnr = compute_psnr(image, self.clean_img)
        hamming_loss = np.sum(current_output != target_output) / len(current_output)

        if hamming_loss < self.best_hamming_loss:
        # if hamming_loss <= self.best_hamming_loss:
            self.best_adv = image
            self.best_hamming_loss = hamming_loss
            self.best_psnr = psnr
            self.best_lambda = const
            self.best_iteration = kwargs['num_iteration']

        return is_adv, psnr, hamming_loss

    def run_attack(self, clean_img, target_output, lr=1e-2, nbs=10, constant_lambda=10**5, max_iteration=1000, confidence=0,**kwargs):
        """

        :param clean_img:
        :param target_output: use {-1,1} instead of {0,1} to better change the logits
        :param lr:
        :param nbs:
        :param constant_lambda:
        :param max_iteration:
        :param confidence:
        :param kwargs:
        :return:
        """
        # initial records
        self.clean_img = clean_img
        self.target_output = target_output
        self.best_adv = None
        self.best_psnr = -np.inf
        self.best_hamming_loss = np.inf
        self.best_iteration = 0
        self.best_lambda = 0
        self.start_t = kwargs['start_t']
        min_, max_ = (0, 1)

        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_img.shape)

        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(w):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            w = np.tanh(w)

            grad = 1 - np.square(w)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            w = w * b + a

            grad = grad * b
            return w, grad

        att_clean_img = to_attack_space(clean_img)
        reconstructed_clean_img, _ = to_model_space(att_clean_img)

        confidence = confidence * np.ones_like(target_output)
        const = constant_lambda
        lower_bound = 0
        upper_bound = np.inf
        for binary_search_step in range(nbs):
            att_perturbation = np.zeros_like(att_clean_img)
            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf
            for iteration in range(max_iteration):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)

                # do one forward to get ecoc logits and find how who should minus whom
                logits = self.predict_logits(perturbed)

                is_adv, linf, psnr = self.new_perturbed(image=perturbed, target_output=target_output,
                                                        constant=const, num_iteration=iteration,
                                                        conf=confidence)

                '''overall loss'''
                cwloss = self.cw_loss(perturbed, const, target_output, confidence)
                cwgrad = self.cw_loss_grad(perturbed, const, target_output, confidence)
                # L2 loss
                s = max_ - min_
                squared_l2_distance = np.sum((perturbed - reconstructed_clean_img) ** 2) / s ** 2
                squared_l2_distance_grad = (2 / s ** 2) * (perturbed - reconstructed_clean_img)

                total_loss = squared_l2_distance + cwloss
                total_grad = squared_l2_distance_grad + cwgrad

                model_space_loss, model_space_grad = total_loss, total_grad

                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grad * dpertub_dw

                gradient_vanished = True if np.sum(att_space_grad) == 0 else False

                if gradient_vanished:
                    assert np.sum(optimizer(att_space_grad, lr)) == 0
                    break
                att_perturbation -= optimizer(att_space_grad, lr)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if iteration % (np.ceil(max_iteration / 10)) == 0:
                    # print('activated output', self.model.predict(perturbed, output_type='activated'))
                    # print('ECOC logits',ecoc_logits)
                    # print('model space loss', model_space_loss)

                    # after each tenth of the iterations, check progress
                    if not (model_space_loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = model_space_loss

            # binary search part
            if found_adv:
                # logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                # logging.info(
                #     "failed to find adversarial " "with const = {}".format(const)
                # )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

            if time.time() - self.start_t > 3600:
                print('time consumption more than 1 hour break')
                break

        return {'adv': self.best_adv, 'psnr': self.best_psnr, 'hamming_loss': self.best_hamming_loss,
                'iteration': self.best_iteration,  'is_full_success': self.best_hamming_loss==0, 'lambda': self.best_lambda}



class MultiLabelCW_act(KerasModelAttack):
    """
    This is my implementation of CW attack for multi-label attack, the attack is the same as the multi-label adversarial
    paper( ICDM conference)
    """
    def __init__(self, model):
        super(MultiLabelCW_act, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax)
        # logits = keras_model.output.op.inputs[0]
        logits = keras_model.output.op.outputs[0]
        # logits_list = tf.unstack(logits, axis=1)
        self._logits_func = K.function([keras_model.input], [logits])

        # ML-CW loss = sum of each bit CW loss
        constant_lambda = K.placeholder(shape=(None, 1))
        target_output = K.placeholder(shape=keras_model.output_shape[1])
        # target_outputs = tf.unstack(target_output)
        confidence = K.placeholder(shape=keras_model.output_shape[1])
        zeros = tf.constant(np.zeros(keras_model.output_shape[1]),dtype=tf.float32)

        CWloss = tf.reduce_sum(tf.maximum(tf.multiply(-target_output, logits) + confidence, zeros))
        # CWloss = constant_lambda * tf.reduce_sum([tf.reduce_max(0, y*l)-conficence for y, l in zip(target_outputs, logits_list)])
        CWloss_grads = K.gradients(CWloss, keras_model.input)
        self._CWloss_fn = K.function([constant_lambda, target_output, keras_model.input, confidence], [CWloss])
        self._CWloss_grads_fn = K.function([constant_lambda, target_output, keras_model.input, confidence], CWloss_grads)

        # TODO: finish all this
    def predict_logits(self, image): return np.squeeze(self._logits_func([image])[0])

    def cw_loss(self, image, constant_lambda, target_output, confidence):
        return self._CWloss_fn([constant_lambda, target_output, image, confidence])[0]

    def cw_loss_grad(self, image, constant_lambda, target_output, confidence):
        return self._CWloss_grads_fn([constant_lambda, target_output, image, confidence])[0]

    def new_perturbed(self, image, target_output, **kwargs):
        # ML model by default has sigmoid activation
        # map prediction to {-1,1}
        const = kwargs['constant']
        current_output = np.sign(np.squeeze(self.model.predict(image)) - 0.5)
        is_adv = np.array_equal(current_output, target_output)
        psnr = compute_psnr(image, self.clean_img)
        hamming_loss = np.sum(current_output != target_output) / len(current_output)

        if hamming_loss < self.best_hamming_loss:
        # if hamming_loss <= self.best_hamming_loss:
            self.best_adv = image
            self.best_hamming_loss = hamming_loss
            self.best_psnr = psnr
            self.best_lambda = const
            self.best_iteration = kwargs['num_iteration']

        return is_adv, psnr, hamming_loss

    def run_attack(self, clean_img, target_output, lr=1e-2, nbs=10, constant_lambda=10**5, max_iteration=1000, confidence=0,**kwargs):
        """

        :param clean_img:
        :param target_output: use {-1,1} instead of {0,1} to better change the logits
        :param lr:
        :param nbs:
        :param constant_lambda:
        :param max_iteration:
        :param confidence:
        :param kwargs:
        :return:
        """
        # initial records
        self.clean_img = clean_img
        self.target_output = target_output
        self.best_adv = None
        self.best_psnr = -np.inf
        self.best_hamming_loss = np.inf
        self.best_iteration = 0
        self.best_lambda = 0
        min_, max_ = (0, 1)

        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_img.shape)

        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(w):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            w = np.tanh(w)

            grad = 1 - np.square(w)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            w = w * b + a

            grad = grad * b
            return w, grad

        att_clean_img = to_attack_space(clean_img)
        reconstructed_clean_img, _ = to_model_space(att_clean_img)

        confidence = confidence * np.ones_like(target_output)
        const = constant_lambda
        lower_bound = 0
        upper_bound = np.inf
        for binary_search_step in range(nbs):
            att_perturbation = np.zeros_like(att_clean_img)
            found_adv = False  # found adv with the current const
            loss_at_previous_check = np.inf
            for iteration in range(max_iteration):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed, dpertub_dw = to_model_space(att_clean_img - att_perturbation)

                # do one forward to get ecoc logits and find how who should minus whom
                logits = self.predict_logits(perturbed)

                is_adv, linf, psnr = self.new_perturbed(image=perturbed, target_output=target_output,
                                                        constant=const, num_iteration=iteration,
                                                        conf=confidence)

                '''overall loss'''
                cwloss = self.cw_loss(perturbed, const, target_output, confidence)
                cwgrad = self.cw_loss_grad(perturbed, const, target_output, confidence)
                # L2 loss
                s = max_ - min_
                squared_l2_distance = np.sum((perturbed - reconstructed_clean_img) ** 2) / s ** 2
                squared_l2_distance_grad = (2 / s ** 2) * (perturbed - reconstructed_clean_img)

                total_loss = squared_l2_distance + cwloss
                total_grad = squared_l2_distance_grad + cwgrad

                model_space_loss, model_space_grad = total_loss, total_grad

                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grad * dpertub_dw

                gradient_vanished = True if np.sum(att_space_grad) == 0 else False

                if gradient_vanished:
                    assert np.sum(optimizer(att_space_grad, lr)) == 0
                    break
                att_perturbation -= optimizer(att_space_grad, lr)

                if is_adv:
                    # this binary search step can be considered a success
                    # but optimization continues to minimize perturbation size
                    found_adv = True

                if iteration % (np.ceil(max_iteration / 10)) == 0:
                    # print('activated output', self.model.predict(perturbed, output_type='activated'))
                    # print('ECOC logits',ecoc_logits)
                    # print('model space loss', model_space_loss)

                    # after each tenth of the iterations, check progress
                    if not (model_space_loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = model_space_loss

            # binary search part
            if found_adv:
                # logging.info("found adversarial with const = {}".format(const))
                upper_bound = const
            else:
                # logging.info(
                #     "failed to find adversarial " "with const = {}".format(const)
                # )
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

        return {'adv': self.best_adv, 'psnr': self.best_psnr, 'hamming_loss': self.best_hamming_loss,
                'iteration': self.best_iteration,  'is_full_success': self.best_hamming_loss==0, 'lambda': self.best_lambda}


class LOTSattack(KerasModelAttack):
    """
    This is my implementation of LOTS attack for one-hot and multi-label model (for LOTS attack they are the same)
    """

    def __init__(self, model):
        super(LOTSattack, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax/sigmoid), i.e. second last layer
        logits = keras_model.output.op.inputs[0]
        self._logits_func = K.function([keras_model.input], [logits])
        target_logits = K.placeholder(shape=(1, logits.shape[1]))
        LOTSloss = K.dot(target_logits - logits, K.transpose(target_logits - logits)) / 2
        LOTSgrad = K.gradients(LOTSloss, keras_model.input)
        self._LOTSloss_fn = K.function([keras_model.input, target_logits], [LOTSloss])
        self._LOTSgrad_fn = K.function([keras_model.input, target_logits], LOTSgrad)

    def predict_logits(self, image): return np.squeeze(self._logits_func([image])[0])

    def LOTS_loss(self, image, target_logits):
        return self._LOTSloss_fn([image, target_logits])[0]

    def LOTS_grad(self, image, target_logits):
        return self._LOTSgrad_fn([image, target_logits])[0]

    def new_perturbed(self, image, target_label, **kwargs):
        # ML model by default has sigmoid activation
        # map prediction to {0,1}
        current_output = (np.sign(np.squeeze(self.model.predict(image)) - 0.5)+1)/2

        is_adv = np.array_equal(current_output, target_label)
        psnr = compute_psnr(image, self.clean_img)
        hamming_loss = np.sum(current_output != target_label) / len(current_output)

        if is_adv and self.best_psnr < psnr:
            self.best_adv = image
            self.best_hamming_loss = hamming_loss
            self.best_psnr = psnr
            self.best_iteration = kwargs['num_iteration']

        if is_adv==False and self.best_loss > kwargs['loss']:
            self.best_loss = kwargs['loss']
            self.best_adv = image
            self.best_iteration = kwargs['num_iteration']
            self.best_hamming_loss = hamming_loss
            self.best_psnr = psnr

        return is_adv, psnr, hamming_loss

    def run_attack(self, clean_img, target_output, target_imgs=None,
                   target_logits=None, **kwargs):
        '''

        :param clean_img:
        :param target_output: in {0,1}
        :param target_imgs:
        :param target_logits:
        :param kwargs:
        :return:
        '''
        # initial records
        self.clean_img = clean_img
        self.target_output = target_output
        self.best_adv = None
        self.best_psnr = -np.inf
        self.best_hamming_loss = np.inf
        self.best_iteration = 0
        self.best_loss = np.inf

        # assert target_imgs is not None or target_output is not None, 'either target images or logits is needed'
        if target_logits is None:
            assert target_imgs is not None, 'either target images or logits is needed'
            target_logits = np.mean(self.predict_logits(target_imgs), axis=0)

        try: max_iter = kwargs['max_iteration']
        except: max_iter = 2000

        perturbed  = clean_img
        found_adv = False

        for i in range(max_iter):
            loss = self.LOTS_loss(perturbed, target_logits)
            gradient = self.LOTS_grad(perturbed, target_logits)
            gradient = gradient / np.max(np.abs(gradient))

            perturbed = (np.round(perturbed * 255) - gradient) / 255
            perturbed = np.clip(perturbed, 0, 1)

            is_adv, psnr, hamming_loss = self.new_perturbed(perturbed, target_output, num_iteration=i, loss=loss)
            if is_adv:
                found_adv = True
            else:
                found_adv = False


        return {'adv': self.best_adv, 'psnr': self.best_psnr, 'hamming_loss': self.best_hamming_loss,
                'iteration': self.best_iteration,  'is_full_success': found_adv}




""" 
From here the blow codes are for binary relevance models, which are a bunch of binary classification models that together
solves a mulit-label classification problem, codes here is only for flying test, the results are never used.
"""

class BinaryRelevanceModel(object):
    """
    t
    """
    def __init__(self, model_list: list):
        # self.models = model_list
        self.N = len(model_list)
        common_input = tf.keras.Input(shape=model_list[0].input_shape[1:])
        # output_list = [model(common_input) for model in model_list]
        output_list = []
        for idx,model in enumerate(model_list):
            x = common_input
            for layer in model.layers:
                layer._name = 'bit{}_'.format(idx+1) + layer.name
                x = layer(x)
            output_list.append(x)

        self.model = tfkerasModel(inputs=common_input, outputs=tf.keras.layers.Concatenate()(output_list))

        logits_list = [output.op.inputs[0] for output in output_list]
        jacobian_matrix_list_tf = [tf.gradients(y, common_input) for y in logits_list]


        logits = tf.concat(logits_list, 1)
        jacobian_matrix = tf.concat(jacobian_matrix_list_tf, name='jacobian_matrix', axis=0)

        self.predict_fn = K.function([common_input], [self.model.output])
        self.logits_fn = K.function([common_input], [logits])
        self.jacobian_fn = K.function([common_input], [jacobian_matrix])

    def predict_output(self, inputs):
        return self.predict_fn([inputs])[0]

    def predict_logits(self, inputs):
        return self.logits_fn([inputs])[0]

    def predict_jacobian(self, inputs):
        return np.squeeze(self.jacobian_fn(inputs)[0])


class JacbMDisTryAndOptimizeBR(BinaryRelevanceModel):
    def __init__(self, model_list):
        super(JacbMDisTryAndOptimizeBR,self).__init__(model_list)

    def new_perturbed(self, perturbeds,  target_output, select_best_dis = False, current_logits = None):
        # Aug 24: modifying for VOC Multilable
        # if current_logits is None:
        #     current_logits = self._logits_func([perturbeds])[0]
        # current_label = np.argmax(current_logits)
        # target_label = np.argmax(target_output)
        # is_adv = np.equal(current_label, target_label)

        # given target_ouput is one-hot or multi-hot vector
        # final_activation_type = self.model.layers[-1].get_config()['activation'] # strs in common activation functions
        try:
            final_activation_type = self.kwargs['activation_type']
        except KeyError:
            final_activation_type = 'sigmoid'
        predicts = np.squeeze(self.model.predict(perturbeds))
        if final_activation_type == 'sigmoid':
            current_output = (1 + np.sign(np.squeeze(self.model.predict(perturbeds)) - 0.5)) / 2
        elif final_activation_type == 'softmax':
            current_output = np.argmax(predicts)
        else:
            raise NotImplementedError('An activation function other than sigmoid/softmax detected, metric of valid adv shall be specified.')
        is_adv = np.array_equal(current_output, target_output)
        psnr = compute_psnr(perturbeds, self.clean_image)
        if select_best_dis:
            if psnr > self.best_psnr:
                self.best_adv = perturbeds
                self.best_psnr = psnr
        return is_adv, psnr

    def run_attack(self, clean_img, targeting_output, **kwargs):
        # ini record
        self.clean_image = clean_img
        self.target_output = targeting_output
        self.best_psnr = 0
        self.best_adv = None
        self.best_iter = 0
        self.best_logits = None
        self.kwargs = kwargs

        def Pq_solver(P, q, G=None, h=None, A=None, b=None):
            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P)
            q = matrix(q)
            if G is not None:
                G = matrix(G)
            if h is not None:
                h = matrix(h)
            if A is not None:
                A = matrix(A)
            if b is not None:
                b = matrix(b)
            solvers.options['show_progress'] = False
            try:
                # sol = solvers.qp(P, q, G, h, A, b)
                sol = solvers.qp(P, q, G, h, A, b, solver='mosek')
                # print('diff solver equal = ', np.array_equal(sol['x'], sol_mosek['x']))
            except ValueError:
                sol = None

            return sol

        def generate_perturbetion(target_output, logit_P, J_matrix, confidence):
            # transform target_output to {-1,1}
            target_output = 2 * target_output - 1
            assert 0 not in target_output, '0 detected in target_output: {}'.format(target_output)

            '''copy fromm new attak'''
            logit_P = np.squeeze(logit_P)
            n = len(logit_P)
            A = -np.array(target_output) * np.eye(n)
            b = np.array(target_output) * logit_P
            # if apply confidence
            b = b - confidence * np.ones_like(b)
            solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
                                 q=b, G=-np.eye(n), h=np.zeros(n))

            # target_label = np.argmax(
            #     np.matmul(target_output, self.output_codebook.T))  # denoted as t in the report
            # clean_label = np.argmax(np.matmul(logit_P, self.output_codebook))
            #
            # _n = list(range(n))
            # _n.remove(target_label)
            # A = np.array([self.output_codebook[i] - self.output_codebook[target_label] for i in _n])
            # if A.shape != (n - 1, n):
            #     A = A.transpose()
            #     assert A.shape == (n - 1, n)
            # b = gtls = np.array([np.matmul(logit_P, self.output_codebook[target_label])
            #                      - np.matmul(logit_P, self.output_codebook[l]) for l in _n])
            # # if apply confidence
            # b = b - confidence * np.ones_like(b)
            # solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
            #                      q=b, G=-np.eye(n - 1), h=np.zeros(n - 1))
            lambda_star = np.array(solution['x'])
            # cases when no solution found for Pq problem
            if solution['x'] is None:
                lambda_star = np.zeros((20, 1))
            d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
            perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
            return perturbetion_x_star, d_star, lambda_star


        try:
            confidence = kwargs['confidence']
        except KeyError:
            confidence = 0

        step_size = kwargs['step_size']
        max_iteration = kwargs['max_iteration']
        verbos = kwargs['verbos']
        try:
            debuging_record_file = kwargs['record_file']
        except KeyError:
            debuging_record_file = None

        with open(debuging_record_file, 'a') as f:
            f.write('{}'.format(kwargs))

        before_perturb_img = clean_img
        J_rank_ob = []  # eventually of shape iters * 4
        clean_logits = self.predict_logits(clean_img)
        for iteration in range(max_iteration):

            # new iter img = last iter perturbed

            # caculate the first perturbetion and check
            before_perturb_logits = self.predict_logits(before_perturb_img)
            Jacobian_J = self.predict_jacobian(before_perturb_img)  # v1.1: derivative at logits level
            J_matrix = Jacobian_J.reshape(len(targeting_output), np.prod(clean_img.shape))

            # if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
            #     # no advs can be found, attack is invalid
            #     is_adv, psnr = new_perturbed(perturbed,  targeting_codeword = targeting_codeword,)
            #     return is_adv, perturbed, psnr
            # check the rank of J_matrix
            J_rank = int(np.linalg.matrix_rank(J_matrix))
            # J_ranks.append(J_rank)
            # check relationship between low rank and out of bound pixels

            perturbetion, d_star, lambda_star = generate_perturbetion(targeting_output, before_perturb_logits, J_matrix,
                                                                        confidence)
            ''' double-checking the generated perturbetion'''
            p_norm = np.sqrt(np.sum(np.square(perturbetion)))
            mse_Jp_dstar = np.mean(np.square(np.matmul(J_matrix, perturbetion) - d_star))
            obs = np.array([np.sum(perturbetion > x)/np.prod(perturbetion.shape) for x in (1,10,100)])
            J_rank_ob.append(np.concatenate(([J_rank], obs)))
            # if max(abs(perturbetion)) > 100:
            #     print('somthing')

            perturbetion = perturbetion.reshape(clean_img.shape)

            after_perturb_img =  np.clip(before_perturb_img + perturbetion, 0, 1)
            is_adv, psnr = self.new_perturbed(after_perturb_img, targeting_output)
            after_perturb_logits = self.predict_logits(np.clip(before_perturb_img + perturbetion, 0, 1))

            p_dir = perturbetion / p_norm

            if is_adv:
                # bs search and return
                upper = 1
                under = 0
                while upper - under > 0.01:
                    # number of bs searches
                    mid = (upper + under) / 2
                    perturbetion = mid * p_dir * p_norm
                    after_perturb_img = np.clip(before_perturb_img + perturbetion, 0, 1)
                    is_adv, psnr = self.new_perturbed(after_perturb_img, targeting_output, select_best_dis=True)
                    if is_adv:
                        # lower the upper
                        upper = mid
                    else:
                        under = mid
                return self.best_adv, self.best_psnr, iteration, J_rank_ob
            else:
                # move the perturbed a bit and test again
                before_perturb_img = np.clip(before_perturb_img + step_size * p_dir, 0, 1)
        # all fails
            if verbos:
                # print iterative debugging information
                # Jrank, logits changes, targeting output
                print('\n========== iteration {} =============\n'
                      'norms and ranks\n'
                      '        Jacobian matrix rank = {}\n'
                      '        perturbetion L2: {}\n'
                      '        mse<J*per, d_star>: {}\n'
                      '        PSNR = {}\n'
                      '        outrange percentage = {}'.format(iteration,
                                                   J_rank,
                                                   p_norm,
                                                   mse_Jp_dstar,
                                                   psnr,
                                                   obs))
                print('\nlogits change'
                      '\n        before perturbetion: {}'
                      '\n        after perturbetion : {}'
                      '\n        targeting output:    {}'
                      '\n        clean logits:        {}'
                            .format(list(before_perturb_logits.round(4)),
                                    list(after_perturb_logits.round(4)),
                                    list(targeting_output.round(0)),
                                    list(clean_logits.round(4))))
            if debuging_record_file:
                with open(debuging_record_file,  'a') as f:
                    f.write('\n========== iteration {} =============\n'
                      'norms and ranks\n'
                      '        Jacobian matrix rank = {}\n'
                      '        perturbetion L2: {}\n'
                      '        mse<J*per, d_star>: {}\n'
                      '        PSNR = {}\n'
                      '        outrange percentage = {}'.format(iteration,
                                                   J_rank,
                                                   p_norm,
                                                   mse_Jp_dstar,
                                                   psnr,
                                                   obs))
                    f.write('\nlogits change'
                      '\n        before perturbetion: {}'
                      '\n        after perturbetion : {}'
                      '\n        targeting output:    {}'
                      '\n        clean logits:        {}'
                            .format(list(before_perturb_logits.round(4)),
                                    list(after_perturb_logits.round(4)),
                                    list(targeting_output.round(0)),
                                    list(clean_logits.round(4))))
        return None, 0, 0, J_rank_ob


class BinaryRevelanceModel_debug(object):
    def __init__(self, model_list: list):
        # self.models = model_list
        self.N = len(model_list)
        common_input = tf.keras.Input(shape=model_list[0].input_shape[1:])
        # output_list = [model(common_input) for model in model_list]
        output_list = []
        for idx,model in enumerate(model_list):
            x = common_input
            for layer in model.layers:
                layer._name = 'bit{}_'.format(idx+1) + layer.name
                x = layer(x)
            output_list.append(x)

        self.model = tfkerasModel(inputs=common_input, outputs=tf.keras.layers.Concatenate()(output_list))

        logits_list = [output.op.inputs[0] for output in output_list]
        jacobian_matrix_list_tf = [tf.gradients(y, common_input) for y in logits_list]


        logits = tf.concat(logits_list, 1)
        jacobian_matrix = tf.concat(jacobian_matrix_list_tf, name='jacobian_matrix', axis=0)

        self.predict_fn = K.function([common_input], [self.model.output])
        self.logits_fn = K.function([common_input], [logits])
        self.jacobian_fn = K.function([common_input], [jacobian_matrix])

    def predict_output(self, inputs):
        return self.predict_fn([inputs])[0]

    def predict_logits(self, inputs):
        return self.logits_fn([inputs])[0]

    def predict_jacobian(self, inputs):
        return np.squeeze(self.jacobian_fn(inputs)[0])


class JacbMDisTryAndOptimizeBR_debug(BinaryRevelanceModel_debug):
    def __init__(self, model_list):
        super(JacbMDisTryAndOptimizeBR_debug, self).__init__(model_list)

    def new_perturbed(self, perturbeds,  target_output, select_best_dis=False, current_logits=None):
        # Aug 24: modifying for VOC Multilable
        # if current_logits is None:
        #     current_logits = self._logits_func([perturbeds])[0]
        # current_label = np.argmax(current_logits)
        # target_label = np.argmax(target_output)
        # is_adv = np.equal(current_label, target_label)

        # given target_ouput is one-hot or multi-hot vector
        # final_activation_type = self.model.layers[-1].get_config()['activation'] # strs in common activation functions
        try:
            final_activation_type = self.kwargs['activation_type']
        except KeyError:
            final_activation_type = 'sigmoid'
        predicts = np.squeeze(self.model.predict(perturbeds))
        if final_activation_type == 'sigmoid':
            # map sigmoid to {0,1}
            # current_output = (1 + np.sign(np.squeeze(self.model.predict(perturbeds)) - 0.5)) / 2
            # map sigmoid to {-1,1}
            current_output = np.sign(np.squeeze(self.model.predict(perturbeds)) - 0.5)
        elif final_activation_type == 'softmax':
            current_output = np.argmax(predicts)
        else:
            raise NotImplementedError('An activation function other than sigmoid/softmax detected, metric of valid adv shall be specified.')
        is_adv = np.array_equal(current_output, target_output)
        psnr = compute_psnr(perturbeds, self.clean_image)
        if select_best_dis:
            if psnr > self.best_psnr:
                self.best_adv = perturbeds
                self.best_psnr = psnr
        return is_adv, psnr

    def run_attack(self, clean_img, targeting_output, **kwargs):
        # ini record
        self.clean_image = clean_img
        self.target_output = targeting_output
        self.best_psnr = 0
        self.best_adv = None
        self.best_iter = 0
        self.best_logits = None
        self.kwargs = kwargs

        def Pq_solver(P, q, G=None, h=None, A=None, b=None):

            """
            Solves a quadratic program

                minimize    (1/2)*x'*P*x + q'*x
                subject to  G*x <= h
                            A*x = b.


            Input arguments.

                P is a n x n dense or sparse 'd' matrix with the lower triangular
                part of P stored in the lower triangle.  Must be positive
                semidefinite.

                q is an n x 1 dense 'd' matrix.

                G is an m x n dense or sparse 'd' matrix.

                h is an m x 1 dense 'd' matrix.

                A is a p x n dense or sparse 'd' matrix.

                b is a p x 1 dense 'd' matrix or None.

                solver is None or 'mosek'.

                The default values for G, h, A and b are empty matrices with
                zero rows.
            """
            P = matrix(P.astype(np.double))
            q = matrix(q.astype(np.double))
            if G is not None:
                G = matrix(G.astype(np.double))
            if h is not None:
                h = matrix(h.astype(np.double))
            if A is not None:
                A = matrix(A.astype(np.double))
            if b is not None:
                b = matrix(b.astype(np.double))
            solvers.options['show_progress'] = False
            from mosek import iparam
            solvers.options['MOSEK'] = {iparam.log: 0}
            try:
                # sol = solvers.qp(P, q, G, h, A, b)
                sol = solvers.qp(P, q, G, h, A, b, solver='mosek')
                # print('diff solver equal = ', np.array_equal(sol['x'], sol_mosek['x']))
            except ValueError:
                sol = None

            return sol

        def generate_perturbetion(target_output, logit_P, J_matrix, confidence):
            # transform target_output to {-1,1}
            target_output = 2 * target_output - 1
            assert 0 not in target_output, '0 detected in target_output: {}'.format(target_output)

            '''copy fromm new attak'''
            logit_P = np.squeeze(logit_P)
            n = len(logit_P)
            A = -np.array(target_output) * np.eye(n)
            b = np.array(target_output) * logit_P
            # if apply confidence
            b = b - confidence * np.ones_like(b)
            solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
                                 q=b, G=-np.eye(n), h=np.zeros(n))

            # target_label = np.argmax(
            #     np.matmul(target_output, self.output_codebook.T))  # denoted as t in the report
            # clean_label = np.argmax(np.matmul(logit_P, self.output_codebook))
            #
            # _n = list(range(n))
            # _n.remove(target_label)
            # A = np.array([self.output_codebook[i] - self.output_codebook[target_label] for i in _n])
            # if A.shape != (n - 1, n):
            #     A = A.transpose()
            #     assert A.shape == (n - 1, n)
            # b = gtls = np.array([np.matmul(logit_P, self.output_codebook[target_label])
            #                      - np.matmul(logit_P, self.output_codebook[l]) for l in _n])
            # # if apply confidence
            # b = b - confidence * np.ones_like(b)
            # solution = Pq_solver(P=np.matmul(np.matmul(A, J_matrix), np.matmul(J_matrix.transpose(), A.transpose())),
            #                      q=b, G=-np.eye(n - 1), h=np.zeros(n - 1))
            lambda_star = np.array(solution['x'])
            # cases when no solution found for Pq problem
            if solution['x'] is None:
                lambda_star = np.zeros((20, 1))
            d_star = np.matmul(- np.matmul(J_matrix, J_matrix.transpose()), np.matmul(A.transpose(), lambda_star))
            perturbetion_x_star = - np.matmul(J_matrix.transpose(), np.matmul(A.transpose(), lambda_star))
            return perturbetion_x_star, d_star, lambda_star


        try:
            confidence = kwargs['confidence']
        except KeyError:
            confidence = 0

        step_size = kwargs['step_size']
        max_iteration = kwargs['max_iteration']
        verbos = kwargs['verbos']
        try:
            debuging_record_file = kwargs['record_file']
        except KeyError:
            debuging_record_file = None

        with open(debuging_record_file, 'a') as f:
            f.write('{}'.format(kwargs))

        before_perturb_img = clean_img
        J_rank_ob = []  # eventually of shape iters * 4
        clean_logits = self.predict_logits(clean_img)
        for iteration in range(max_iteration):

            # new iter img = last iter perturbed

            # caculate the first perturbetion and check
            before_perturb_logits = self.predict_logits(before_perturb_img)
            Jacobian_J = self.predict_jacobian(before_perturb_img)  # v1.1: derivative at logits level
            J_matrix = Jacobian_J.reshape(len(targeting_output), np.prod(clean_img.shape))

            # if np.sum(np.square(Jacobian_J)) == 0:  # somehow Jacobian is 0
            #     # no advs can be found, attack is invalid
            #     is_adv, psnr = new_perturbed(perturbed,  targeting_codeword = targeting_codeword,)
            #     return is_adv, perturbed, psnr
            # check the rank of J_matrix
            J_rank = int(np.linalg.matrix_rank(J_matrix))
            # J_ranks.append(J_rank)
            # check relationship between low rank and out of bound pixels

            perturbetion, d_star, lambda_star = generate_perturbetion(targeting_output, before_perturb_logits, J_matrix,
                                                                        confidence)
            ''' double-checking the generated perturbetion'''
            p_norm = np.sqrt(np.sum(np.square(perturbetion)))
            mse_Jp_dstar = np.mean(np.square(np.matmul(J_matrix, perturbetion) - d_star))
            obs = np.array([np.sum(perturbetion > x)/np.prod(perturbetion.shape) for x in (1,10,100)])
            J_rank_ob.append(np.concatenate(([J_rank], obs)))
            # if max(abs(perturbetion)) > 100:
            #     print('somthing')

            perturbetion = perturbetion.reshape(clean_img.shape)

            after_perturb_img =  np.clip(before_perturb_img + perturbetion, 0, 1)
            is_adv, psnr = self.new_perturbed(after_perturb_img, targeting_output)
            after_perturb_logits = self.predict_logits(np.clip(before_perturb_img + perturbetion, 0, 1))

            p_dir = perturbetion / p_norm

            if is_adv:
                # bs search and return
                upper = 1
                under = 0
                while upper - under > 0.01:
                    # number of bs searches
                    mid = (upper + under) / 2
                    perturbetion = mid * p_dir * p_norm
                    after_perturb_img = np.clip(before_perturb_img + perturbetion, 0, 1)
                    is_adv, psnr = self.new_perturbed(after_perturb_img, targeting_output, select_best_dis=True)
                    if is_adv:
                        # lower the upper
                        upper = mid
                    else:
                        under = mid
                return self.best_adv, self.best_psnr, iteration, J_rank_ob
            else:
                # move the perturbed a bit and test again
                before_perturb_img = np.clip(before_perturb_img + step_size * p_dir, 0, 1)
        # all fails
            if verbos:
                # print iterative debugging information
                # Jrank, logits changes, targeting output
                print('\n========== iteration {} =============\n'
                      'norms and ranks\n'
                      '        Jacobian matrix rank = {}\n'
                      '        perturbetion L2: {}\n'
                      '        mse<J*per, d_star>: {}\n'
                      '        PSNR = {}\n'
                      '        outrange percentage = {}'.format(iteration,
                                                   J_rank,
                                                   p_norm,
                                                   mse_Jp_dstar,
                                                   psnr,
                                                   obs))
                print('\nlogits change'
                      '\n        before perturbetion: {}'
                      '\n        after perturbetion : {}'
                      '\n        targeting output:    {}'
                      '\n        clean logits:        {}'
                            .format(list(before_perturb_logits.round(4)),
                                    list(after_perturb_logits.round(4)),
                                    list(targeting_output.round(0)),
                                    list(clean_logits.round(4))))
            if debuging_record_file:
                with open(debuging_record_file,  'a') as f:
                    f.write('\n========== iteration {} =============\n'
                      'norms and ranks\n'
                      '        Jacobian matrix rank = {}\n'
                      '        perturbetion L2: {}\n'
                      '        mse<J*per, d_star>: {}\n'
                      '        PSNR = {}\n'
                      '        outrange percentage = {}'.format(iteration,
                                                   J_rank,
                                                   p_norm,
                                                   mse_Jp_dstar,
                                                   psnr,
                                                   obs))
                    f.write('\nlogits change'
                      '\n        before perturbetion: {}'
                      '\n        after perturbetion : {}'
                      '\n        targeting output:    {}'
                      '\n        clean logits:        {}'
                            .format(list(before_perturb_logits.round(4)),
                                    list(after_perturb_logits.round(4)),
                                    list(targeting_output.round(0)),
                                    list(clean_logits.round(4))))
        return None, 0, 0, J_rank_ob
