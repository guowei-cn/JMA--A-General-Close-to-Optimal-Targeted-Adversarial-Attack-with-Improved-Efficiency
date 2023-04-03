import sys
import tensorflow as tf

from core_attack.Attacks import KerasModelAttack

tf.compat.v1.disable_eager_execution()
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Model
from collections import namedtuple
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.math_func import compute_psnr
from scipy import optimize
from cvxopt import matrix
from cvxopt import solvers
from scipy.special import softmax


class ECOCAttack(object):

    def __init__(self, ecoc_model):
        if hasattr(ecoc_model, 'keras_model'):
            self.model = ecoc_model
            keras_model = ecoc_model.keras_model
            # keras model is just for backend opreation, while LatentModel instance is for prediction and other stuff
            assert isinstance(keras_model, Model)
        else:
            raise TypeError('Input model must be ethier customed wrapped model')


class CWBAttack(ECOCAttack):
    def __init__(self, ecoc_model):
        super(CWBAttack, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model
        # activated output level

        y_trues = K.placeholder(shape=(None, keras_model.output_shape[1]))
        individual_activations = keras_model.layers[-1].input
        logits = tf.concat([activate.op.inputs[0] for activate in individual_activations], axis=1)
        confidence_thres = K.placeholder(shape=(1))
        # thres been expand with a new axis to be aglined with logits, do this when call gradient func
        confidence_losses = tf.minimum(K.min(2 * logits * y_trues , axis=1), confidence_thres)
        # confidence_losses = K.min([z * 2 * y_trues[i] for i, z in enumerate(logits)] + [confidence_thres], axis=-1)
        self._confidence_losses_func = K.function([keras_model.input, y_trues, confidence_thres], [confidence_losses])
        dcdx = K.gradients(confidence_losses, keras_model.input)  # derivative of confidence over image input x
        self._confidence_gradient_func = K.function([keras_model.input, y_trues, confidence_thres], dcdx)
        self._logits_func = K.function([keras_model.input], [logits])

    def run_attack(self, clean_images, targeting_codewords, confidence = 0, initial_constant=1e-3, learning_rate=1e-2,
                       num_binary_search=9, max_iterations=10000):

        """

        :param clean_img:
        :param confidence:
        :param initial_constant:
        :param learning_rate:
        :param num_binary_search:
        :param max_iterations:
        :param targeting_un_decoded: {-1, +1} obtained directly by sign function
        :return:
        """
        # variables will be used for whole attack
        self.batch_size = clean_images.shape[0]
        self.clean_images = clean_images
        self.targeting_codewords = targeting_codewords
        self.confidence_thres = confidence
        min_, max_ = (0, 1)

        # adam algorithm for gradient updating
        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_images.shape)

        #records for return
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(clean_images.shape[0])
        self.best_iters = np.zeros(clean_images.shape[0])
        self.best_consts = np.zeros(clean_images.shape[0])
        self.best_l2s = np.inf * np.ones(clean_images.shape[0])
        self.best_linfs = np.inf * np.ones(clean_images.shape[0])
        self.best_confidences = np.zeros(self.batch_size)
        self.best_bs_nums = np.zeros(self.batch_size)

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

        def overall_loss_grad(imgs, reconstruct_clean_imgs, target_codewords, constant, confidence):
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty

            confidence = np.array(confidence)
            # while confidence.ndim < 2:
            #     confidence = confidence[np.newaxis]

            confidence_losses = self._confidence_losses_func([imgs, target_codewords, confidence])[
                0]  # this loss should be maxmized to get desired error
            # print('confidence loss:', confidence_loss)

            confidence_grads = self._confidence_gradient_func([imgs, target_codewords, confidence])[0]

            # L2 distance loss, grad w.r.t. img
            squared_l2_distances = np.sum((imgs - reconstruct_clean_imgs) ** 2, axis=(1,2,3)) / s ** 2
            squared_l2_distance_grads = (2 / s ** 2) * (imgs - reconstruct_clean_imgs)

            # total loss, grad w.r.t. img
            overall_loss = squared_l2_distances - constant * confidence_losses
            overall_grad = squared_l2_distance_grads - np.array([c * cg for c,cg in zip(constant, confidence_grads)])
            return overall_loss, overall_grad

        def new_perturbed(perturbed_images, current_iters, current_constants, current_bs_num, strict_conf_thres = True):
            current_logits = np.squeeze(np.array(self._logits_func([perturbed_images])))
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            current_codewords = self.model.hadamard_matrix[current_labels]
            current_confidences = np.min(2* self.targeting_codewords * current_logits, axis=1) # should be bit-wise multiply
            # 2 * targeting_codewords * current_logits

            is_advs = np.prod(current_codewords==self.targeting_codewords, axis=1)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))
                if strict_conf_thres:
                    if is_advs[index] and psnr > self.best_distances[index] and current_confidences[index] >= self.confidence_thres:
                        self.best_distances[index] = psnr
                        self.best_advs[index] = perturbed
                        self.best_consts[index] = current_constants[index]
                        self.best_iters[index] = current_iters
                        self.best_l2s[index] = l2
                        self.best_linfs[index] = linf
                        self.best_confidences[index] = current_confidences[index]
                        self.best_bs_nums[index] = current_bs_num
                else:
                    if is_advs[index] and psnr > self.best_distances[index]:
                        self.best_distances[index] = psnr
                        self.best_advs[index] = perturbed
                        self.best_consts[index] = current_constants[index]
                        self.best_iters[index] = current_iters
                        self.best_l2s[index] = l2
                        self.best_linfs[index] = linf
                        self.best_confidences[index] = current_confidences[index]
                        self.best_bs_nums[index] = current_bs_num

            return is_advs, self.best_advs, self.best_distances, self.best_iters, self.best_consts, \
                   self.best_confidences, self.best_bs_nums



        # variables representing inputs in attack space will be
        # prefixed with att_
        att_clean_images = to_attack_space(clean_images)
        # will be close but not identical to clean_img
        reconstructed_clean_images, _ = to_model_space(att_clean_images)

        # binary search for good constant c
        # the binary search finds the smallest const for which we
        # find an adversarial
        consts = initial_constant * np.ones(self.batch_size)
        lower_bounds = np.zeros(self.batch_size)
        upper_bound = np.inf * np.ones(self.batch_size)
        for binary_search_step in range(num_binary_search):
            att_perturbations = np.zeros_like(att_clean_images)
            found_advs = [False] * self.batch_size  # found adv with the current const
            losses_at_previous_check = np.inf  # [np.inf] * self.batch_size
            for iteration in range(max_iterations):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed_images, dpertub_dw = to_model_space(att_clean_images - att_perturbations)
                # # compute current confidence
                # current_logits = np.squeeze(np.array(self._logits_func([perturbed]))).T
                # bit_confidences = 2 * targeting_codewords * current_logits
                # current_confidence = np.min(bit_confidences, axis=1)

                is_advs, best_advs, best_distances, best_iters, best_consts, best_confidences, best_bs_nums\
                    = new_perturbed(perturbed_images, iteration, consts, binary_search_step)
                # is_adv, linf, per_predictions, per_predict_codeword = self._is_adv(reconstructed_clean_img, perturbed,
                #                                                                    target_code)
                model_space_losses, model_space_grads = overall_loss_grad(perturbed_images, reconstructed_clean_images,
                                                                       targeting_codewords,
                                                                       consts, confidence)
                # print('total_loss in model space = %.f', model_space_loss)
                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grads * dpertub_dw # I want it be bit-wise multiply

                att_perturbations -= optimizer(att_space_grad, learning_rate)

                found_advs = np.logical_or(found_advs, is_advs)

                if iteration % (np.ceil(max_iterations / 10)) == 0:

                    # after each tenth of the iterations, check progress
                    if  np.all(model_space_losses >= 0.9999 * losses_at_previous_check):
                        break  # stop Adam if all in batch has no progress
                    losses_at_previous_check = model_space_losses

            # binary search part
            for idx in range(self.batch_size):
                if found_advs[idx]:
                    # success
                    upper_bound[idx] = consts[idx]
                else:
                    # fails
                    lower_bounds[idx] = consts[idx]

                if upper_bound[idx] == np.inf:
                    # exponential search
                    consts[idx] *= 10
                else:
                    # binary search
                    consts[idx] = (lower_bounds[idx] + upper_bound[idx]) / 2

        return self.best_advs, self.best_distances, self.best_iters,\
               self.best_consts, self.best_confidences, self.best_bs_nums

class CWCAttack(ECOCAttack):
    def __init__(self, ecoc_model):
        super(CWCAttack, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model


        hadamard = K.variable(self.model.hadamard_matrix)
        # hadamard = K.variable(self.model.hadamard_matrix.T)

        # logits
        individual_activations = keras_model.layers[-1].input
        logits = tf.concat([activate.op.inputs[0] for activate in individual_activations], axis=1)
        self._logits_func = K.function([keras_model.input], [logits])

        # ecoc_lgits
        ecoc_logits = K.dot(keras_model.output,
                            K.transpose(hadamard))  # shape: (num_img, 32) * (32, 32), hadamard needs transpose in case not square matrix
        self._ecoc_logits_func = K.function([keras_model.input], [ecoc_logits])

        # takes indexes to determinate target and second top class
        confidence_thres = K.placeholder(shape = ())

        class_maximize = K.placeholder(shape=(None,), dtype='int32')
        one_hot_maximize = tf.one_hot(class_maximize, int(ecoc_logits.shape[1]))

        class_minimize = K.placeholder(shape=(None,), dtype='int32')
        one_hot_minimize = tf.one_hot(class_minimize, int(ecoc_logits.shape[1]))

        prod_min = tf.reduce_sum(one_hot_minimize * ecoc_logits, axis=1)
        prod_max = tf.reduce_sum(one_hot_maximize * ecoc_logits, axis=1)

        sub = prod_min - prod_max

        ecoc_confidence_losses = tf.maximum(sub, -confidence_thres)

        self._ecoc_confidence_loss_func = K.function(
            [keras_model.input, class_minimize, class_maximize, confidence_thres], [ecoc_confidence_losses])
        ecoc_confidence_grad = K.gradients(ecoc_confidence_losses, keras_model.input)
        self._ecoc_confidence_grad_fn = K.function(
            [keras_model.input, class_minimize, class_maximize, confidence_thres], ecoc_confidence_grad)

    def run_attack(self, clean_images, targeting_classes, confidence = 0, initial_constant=1e-3, learning_rate=1e-2,
                   num_binary_search=9, max_iterations=10000):
        self.batch_size = clean_images.shape[0]
        self.clean_images = clean_images
        self.confidence_thres = confidence
        self.targeting_classes = targeting_classes

        min_, max_ = (0, 1)

        #records for return
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(clean_images.shape[0])
        self.best_iters = np.zeros(clean_images.shape[0])
        self.best_consts = np.zeros(clean_images.shape[0])
        self.best_l2s = np.inf * np.ones(clean_images.shape[0])
        self.best_linfs = np.inf * np.ones(clean_images.shape[0])
        self.best_confidences = np.zeros(self.batch_size)
        self.best_bs_nums = np.zeros(self.batch_size)

        # adam algorithm for gradient updating
        from foolbox.attacks.carlini_wagner import AdamOptimizer
        optimizer = AdamOptimizer(clean_images.shape)

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

        def overall_loss_grad(imgs, reconstruct_clean_imgs, classes_minimize, classes_maximize, constants, confidence):
            """

            :param imgs:
            :param reconstruct_clean_imgs:
            :param who_minus_whom: # e.g. [ 0, 0, 1, -1, 0, 0]  indicates the third minus the forth
            :param constants:
            :param confidence:
            :return:
            """
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty
            # gradient_vanished = False
            confidence = np.array(confidence)
            # while confidence.ndim < 2:
            #     confidence = confidence[np.newaxis]

            confidence_losses = self._ecoc_confidence_loss_func([imgs, classes_minimize, classes_maximize, confidence])[
                0]  # this loss should be maxmized to get desired error

            confidence_grads = self._ecoc_confidence_grad_fn([imgs, classes_minimize, classes_maximize, confidence])[0]

            # L2 distance loss, grad w.r.t. img
            squared_l2_distances = np.sum((imgs - reconstruct_clean_imgs) ** 2, axis=(1,2,3)) / s ** 2
            squared_l2_distance_grads = (2 / s ** 2) * (imgs - reconstruct_clean_imgs)

            # total loss, grad w.r.t. img
            overall_losses = squared_l2_distances + constants * confidence_losses
            overall_grads = squared_l2_distance_grads + np.array([c * cg for c,cg in zip(constants, confidence_grads)])
            return overall_losses, overall_grads

        def new_perturbed(perturbed_images, current_iters, current_constants, current_bs_num, classes_maximize,
                          classes_minimize, current_ecoc_logits, strict_conf_thres = True):
            current_logits = np.squeeze(np.array(self._logits_func([perturbed_images])))
            # current_ecoc_logits = np.squeeze(np.array(self._ecoc_logits_func([perturbed_images])))
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            # current_labels_ecoc_logits = np.argmax(current_ecoc_logits, axis=1)
            # assert np.array_equal(current_labels, current_labels_ecoc_logits)
            current_confidences = current_ecoc_logits[range(self.batch_size),classes_maximize] \
                                  - current_ecoc_logits[range(self.batch_size),classes_minimize]

            is_advs = self.targeting_classes == current_labels

            # update records inside batch
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))
                if strict_conf_thres:
                    if is_advs[index] and psnr > self.best_distances[index] and\
                            current_confidences[index] >= self.confidence_thres:
                        self.best_distances[index] = psnr
                        self.best_advs[index] = perturbed
                        self.best_consts[index] = current_constants[index]
                        self.best_iters[index] = current_iters
                        self.best_l2s[index] = l2
                        self.best_linfs[index] = linf
                        self.best_confidences[index] = current_confidences[index]
                        self.best_bs_nums[index] = current_bs_num
                else:
                    if is_advs[index] and psnr > self.best_distances[index]:
                        self.best_distances[index] = psnr
                        self.best_advs[index] = perturbed
                        self.best_consts[index] = current_constants[index]
                        self.best_iters[index] = current_iters
                        self.best_l2s[index] = l2
                        self.best_linfs[index] = linf
                        self.best_confidences[index] = current_confidences[index]
                        self.best_bs_nums[index] = current_bs_num

            return is_advs, self.best_advs, self.best_distances, self.best_iters, self.best_consts, \
                   self.best_confidences, self.best_bs_nums

        is_targeted = True  # untargeted not implemented

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_clean_images = to_attack_space(clean_images)
        # will be close but not identical to clean_img
        reconstructed_clean_images, _ = to_model_space(att_clean_images)

        # binary search for good constant c
        # the binary search finds the smallest const for which we
        # find an adversarial
        consts = initial_constant * np.ones(self.batch_size)
        lower_bounds = np.zeros(self.batch_size)
        upper_bound = np.inf * np.ones(self.batch_size)
        for binary_search_step in range(num_binary_search):
            att_perturbations = np.zeros_like(att_clean_images)
            found_advs = [False] * self.batch_size  # found adv with the current const
            losses_at_previous_check = np.inf  # [np.inf] * self.batch_size
            for iteration in range(max_iterations):
                # do optimize here
                # go back to model space to see if is adv
                # perturbed is model space, dpertub_dw is derivative of model space over attack space
                perturbed_images, dpertub_dw = to_model_space(att_clean_images - att_perturbations)

                # find class to min and max according to current ecoc_logits
                current_ecoc_logits = np.squeeze(np.array(self._ecoc_logits_func([perturbed_images])))

                if is_targeted:
                    classes_maximize = targeting_classes
                    x = np.zeros_like(current_ecoc_logits)
                    x[range(self.batch_size), targeting_classes] = np.inf
                    classes_minimize = np.argmax(current_ecoc_logits - x, axis=1)
                else:
                    raise NotImplementedError
                    # suppose there are clean_classes
                    # clean_classes = targeting_classes
                    # x  = np.zeros_like(current_ecoc_logits)
                    # x[range(self.batch_size), clean_classes] = np.inf
                    # classes_maximize = np.argmax(current_ecoc_logits - x, axis=1)

                is_advs, best_advs, best_distances, best_iters, best_consts, best_confidences, best_bs_nums \
                    = new_perturbed(perturbed_images, iteration, consts, binary_search_step, classes_maximize, classes_minimize,
                                    current_ecoc_logits, strict_conf_thres=True)
                # is_adv, linf, per_predictions, per_predict_codeword = self._is_adv(reconstructed_clean_img, perturbed,
                #                                                                    target_code)
                model_space_losses, model_space_grads = overall_loss_grad(perturbed_images, reconstructed_clean_images,
                                                                          classes_minimize, classes_maximize,
                                                                       consts, confidence)
                # print('total_loss in model space = %.f', model_space_loss)
                # final gradient and update by Adam algorithm
                att_space_grad = model_space_grads * dpertub_dw # I want it be bit-wise multiply

                att_perturbations -= optimizer(att_space_grad, learning_rate)

                found_advs = np.logical_or(found_advs, is_advs)

                if iteration % (np.ceil(max_iterations / 10)) == 0:

                    # after each tenth of the iterations, check progress
                    if  np.all(model_space_losses >= 0.9999 * losses_at_previous_check):
                        break  # stop Adam if all in batch has no progress
                    losses_at_previous_check = model_space_losses

            # binary search part
            for idx in range(self.batch_size):
                if found_advs[idx]:
                    # success
                    upper_bound[idx] = consts[idx]
                else:
                    # fails
                    lower_bounds[idx] = consts[idx]

                if upper_bound[idx] == np.inf:
                    # exponential search
                    consts[idx] *= 10
                else:
                    # binary search
                    consts[idx] = (lower_bounds[idx] + upper_bound[idx]) / 2

        return self.best_advs, self.best_distances, self.best_iters,\
               self.best_consts, self.best_confidences, self.best_bs_nums

class LogitsLOTSAttackv2(ECOCAttack):
    """
    difference between V0, V1 and V2 are whether controling step size of LOTS attack, which is not controlled in the original
    LOTS paper

    This V2 version is a combination of V0 and V1, V2 take a specific parameter to switch between V0 and V1
    """
    def __init__(self, ecoc_model):
        super(LogitsLOTSAttackv2, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model

        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self.codeword_length  = len(logits)
        target_logits = K.placeholder(shape=(None,len(logits)))
        logits_tensor = K.concatenate(logits)
        # LOTS_object = K.dot(target_logits - logits_tensor, K.transpose(target_logits - logits_tensor)) / 2
        LOTS_object = K.sum(K.square(target_logits - logits_tensor),axis=1) / 2
        LOTS_grad = K.gradients(LOTS_object, keras_model.input)

        self._LOTS_loss = K.function([keras_model.input, target_logits], [LOTS_object])
        self._logits_func = K.function([keras_model.input], [logits_tensor])
        self._LOTS_grad_func = K.function([keras_model.input, target_logits], LOTS_grad)


    def run_attack(self, clean_images, batch_target_imgs, targeting_codewords,
                   initial_constant, num_binary_search, tar_pool_size = 50, **kwargs):
        '''

        :param clean_images: of dim 4 (batch_size, 64, 64, 3)
        :param batch_target_imgs: of dim 5 (batch_size, target_img_pool_size =50, 64, 64, 3)
        :param targeting_codewords: of dim 2 (batch size, len(codeword)
        :param kwargs: max iters and step size
        :return:
        '''
        min_, max_ = (0, 1)
        def overall_loss_grad(perturbeds, clean_imgs, target_logits, constants):
            """

            :param perturbeds:
            :param clean_imgs:
            :param who_minus_whom: # e.g. [ 0, 0, 1, -1, 0, 0]  indicates the third minus the forth
            :param constants:
            :param confidence:
            :return:
            """
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty
            # gradient_vanished = False

            LOTS_loss = self._LOTS_loss([perturbeds, target_logits])[0]
            LOTS_grads = self._LOTS_grad_func([perturbeds, target_logits])[0]  # of shape (batch size, image shape)

            # L2 distance loss, grad w.r.t. img
            squared_l2_distances = np.sum((perturbeds - clean_imgs) ** 2, axis=(1, 2, 3)) / s ** 2
            squared_l2_distance_grads = (2 / s ** 2) * (perturbeds - clean_imgs)

            # total loss, grad w.r.t. img
            overall_losses = squared_l2_distances + constants * LOTS_loss
            overall_grads = squared_l2_distance_grads + np.array([c * cg for c,cg in zip(constants, LOTS_grads)])
            return overall_losses, overall_grads

        def new_perturbeds(perturbed_images, current_iter):
            current_logits = self._logits_func([perturbed_images])[0]
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            current_codewords = self.model.hadamard_matrix[current_labels]

            is_advs = np.prod(current_codewords == self.targeting_codewords, axis=1)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))

                if is_advs[index] and psnr > self.best_distances[index]:
                    self.best_distances[index] = psnr
                    self.best_advs[index] = perturbed
                    self.best_iters[index] = current_iter
                    self.best_l2s[index] = l2
                    self.best_linfs[index] = linf

            return is_advs, self.best_advs, self.best_distances, self.best_iters



        max_iter = kwargs['max_iteration']
        try:
            l2_normalize = True
            step_size = kwargs['step_size']
        except KeyError:
            l2_normalize = False
            step_size = 1
        self.batch_size = clean_images.shape[0]

        # records for return
        self.targeting_codewords = targeting_codewords
        self.clean_images = clean_images
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(clean_images.shape[0])
        self.best_iters = np.zeros(clean_images.shape[0])
        self.best_consts = np.zeros(clean_images.shape[0])
        self.best_l2s = np.inf * np.ones(clean_images.shape[0])
        self.best_linfs = np.inf * np.ones(clean_images.shape[0])

        # raw_target_logits = self._logits_func([batch_target_imgs])[0]
        raw_target_logits = np.concatenate([self._logits_func([one_target_imgs])[0] for one_target_imgs in batch_target_imgs], axis=0)
        # of shape (batch_size * target_img_pool_size =50 * batch size, len(codeword)

        # compute mean of target logits every pool
        target_logits = []
        for pool_id in range(int(np.ceil(raw_target_logits.shape[0] / tar_pool_size))):
            try:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: (pool_id + 1) * tar_pool_size], axis=0)
                assert len(ave_for_one) == raw_target_logits.shape[1]
            except IndexError:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: ], axis=0)
            target_logits.append(ave_for_one)
        assert len(target_logits) == self.batch_size
        target_logits = np.array(target_logits)
        assert target_logits.shape == (self.batch_size, self.codeword_length)

        perturbeds = clean_images.copy()
        consts = initial_constant * np.ones(self.batch_size)
        lower_bounds = np.zeros(self.batch_size)
        upper_bound = np.inf * np.ones(self.batch_size)
        for binary_search_step in range(num_binary_search):
            found_advs = [False] * self.batch_size  # found adv with the current const
            losses_at_previous_check = np.inf
            for i in range(max_iter):
                # LOTS_grads = self._LOTS_grad_func([perturbeds, target_logits])[0] # of shape (batch size, image shape)
                LOTS_losses, LOTS_grads = overall_loss_grad(perturbeds, clean_images, target_logits, consts)
                # per image-grad normalize
                for idx, grad in enumerate(LOTS_grads):
                    if l2_normalize:
                        grad = step_size * grad / np.sqrt(np.sum(np.square(grad)))
                    else:
                        grad = grad / np.max(np.abs(grad)) / 255

                    perturbeds[idx] = np.clip(perturbeds[idx] - grad, 0, 1)

                # check batch
                is_advs, best_advs, best_distances, best_iters = new_perturbeds(perturbeds, i)
                found_advs = np.logical_or(found_advs, is_advs)
                # if np.all(is_advs): # see if improvements
                #     break
                if i % (np.ceil(max_iter / 10)) == 0:

                    # after each tenth of the iterations, check progress
                    if np.all(LOTS_losses >= 0.9999 * losses_at_previous_check):
                        break  # stop Adam if all in batch has no progress
                    losses_at_previous_check = LOTS_losses

            # binary search part
            for idx in range(self.batch_size):
                if found_advs[idx]:
                    # success
                    upper_bound[idx] = consts[idx]
                else:
                    # fails
                    lower_bounds[idx] = consts[idx]

                if upper_bound[idx] == np.inf:
                    # exponential search
                    consts[idx] *= 10
                else:
                    # binary search
                    consts[idx] = (lower_bounds[idx] + upper_bound[idx]) / 2
        return self.best_advs, self.best_distances, self.best_iters, self.best_linfs

class LogitsLOTSAttackv1(ECOCAttack):
    """
    modified version that controlling step size
    """

    def __init__(self, ecoc_model):
        super(LogitsLOTSAttackv1, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model

        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self.codeword_length  = len(logits)
        target_logits = K.placeholder(shape=(None,len(logits)))
        logits_tensor = K.concatenate(logits)
        # LOTS_object = K.dot(target_logits - logits_tensor, K.transpose(target_logits - logits_tensor)) / 2
        LOTS_object = K.sum(K.square(target_logits - logits_tensor),axis=1) / 2
        LOTS_grad = K.gradients(LOTS_object, keras_model.input)

        self._logits_func = K.function([keras_model.input], [logits_tensor])
        self._LOTS_grad_func = K.function([keras_model.input, target_logits], LOTS_grad)

    def run_attack(self, clean_images, batch_target_imgs, targeting_codewords, tar_pool_size = 50, **kwargs):
        '''

        :param clean_images: of dim 4 (batch_size, 64, 64, 3)
        :param batch_target_imgs: of dim 5 (batch_size, target_img_pool_size =50, 64, 64, 3)
        :param targeting_codewords: of dim 2 (batch size, len(codeword)
        :param kwargs: max iters and step size
        :return:
        '''

        def new_perturbeds(perturbed_images, current_iter):
            current_logits = self._logits_func([perturbed_images])[0]
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            current_codewords = self.model.hadamard_matrix[current_labels]

            is_advs = np.prod(current_codewords == self.targeting_codewords, axis=1)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))

                if is_advs[index] and psnr > self.best_distances[index]:
                    self.best_distances[index] = psnr
                    self.best_advs[index] = perturbed
                    self.best_iters[index] = current_iter
                    self.best_l2s[index] = l2
                    self.best_linfs[index] = linf

            return is_advs, self.best_advs, self.best_distances, self.best_iters



        max_iter = kwargs['max_iteration']
        step_size = kwargs['step_size']

        self.batch_size = clean_images.shape[0]

        # records for return
        self.targeting_codewords = targeting_codewords
        self.clean_images = clean_images
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(clean_images.shape[0])
        self.best_iters = np.zeros(clean_images.shape[0])
        self.best_consts = np.zeros(clean_images.shape[0])
        self.best_l2s = np.inf * np.ones(clean_images.shape[0])
        self.best_linfs = np.inf * np.ones(clean_images.shape[0])

        # raw_target_logits = self._logits_func([batch_target_imgs])[0]
        raw_target_logits = np.concatenate([self._logits_func([one_target_imgs])[0] for one_target_imgs in batch_target_imgs], axis=0)
        # of shape (batch_size * target_img_pool_size =50 * batch size, len(codeword)

        # compute mean of target logits every pool
        target_logits = []
        for pool_id in range(int(np.ceil(raw_target_logits.shape[0] / tar_pool_size))):
            try:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: (pool_id + 1) * tar_pool_size], axis=0)
                assert len(ave_for_one) == raw_target_logits.shape[1]
            except IndexError:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: ], axis=0)
            target_logits.append(ave_for_one)
        assert len(target_logits) == self.batch_size
        target_logits = np.array(target_logits)
        assert target_logits.shape == (self.batch_size, self.codeword_length)

        perturbeds = clean_images.copy()
        for i in range(max_iter):
            LOTS_grads = self._LOTS_grad_func([perturbeds, target_logits])[0] # of shape (batch size, image shape)

            # per image-grad normalize
            for idx, grad in enumerate(LOTS_grads):

                grad = step_size * grad / np.sqrt(np.sum(np.square(grad)))


                perturbeds[idx] = np.clip(perturbeds[idx] - grad, 0, 1)

            # check batch
            is_advs, best_advs, best_distances, best_iters = new_perturbeds(perturbeds, i)
            # if np.all(is_advs): # see if improvements
            #     break
        return self.best_advs, self.best_distances, self.best_iters, self.best_linfs

class LogitsLOTSAttack(ECOCAttack):
    """
    The original LOTS, excatly the same as the LOTS paper
    """
    def __init__(self, ecoc_model):
        super(LogitsLOTSAttack, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model

        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self.logits_length  = len(logits)
        target_logits = K.placeholder(shape=(None,len(logits)))
        logits_tensor = K.concatenate(logits)
        # LOTS_object = K.dot(target_logits - logits_tensor, K.transpose(target_logits - logits_tensor)) / 2
        LOTS_object = K.sum(K.square(target_logits - logits_tensor),axis=1) / 2
        LOTS_grad = K.gradients(LOTS_object, keras_model.input)

        self._LOTS_loss_func = K.function([keras_model.input, target_logits], [LOTS_object])
        self._logits_func = K.function([keras_model.input], [logits_tensor])
        self._LOTS_grad_func = K.function([keras_model.input, target_logits], LOTS_grad)



    def run_attack(self, clean_images, batch_target_imgs, targeting_codewords, tar_pool_size = 50, **kwargs):
        '''

        :param clean_images: of dim 4 (batch_size, 64, 64, 3)
        :param batch_target_imgs: of dim 5 (batch_size, target_img_pool_size =50, 64, 64, 3)
        :param targeting_codewords: of dim 2 (batch size, len(codeword)
        :param kwargs: max iters and step size
        :return:
        '''

        def new_perturbeds(perturbed_images, current_iter):
            current_logits = self._logits_func([perturbed_images])[0]
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            current_codewords = self.model.hadamard_matrix[current_labels]

            is_advs = np.prod(current_codewords == self.targeting_codewords, axis=1)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))

                if is_advs[index] and psnr > self.best_distances[index]:
                    self.best_distances[index] = psnr
                    self.best_advs[index] = perturbed
                    self.best_iters[index] = current_iter
                    self.best_l2s[index] = l2
                    self.best_linfs[index] = linf

            return is_advs, self.best_advs, self.best_distances, self.best_iters



        max_iter = kwargs['max_iteration']

        self.batch_size = clean_images.shape[0]

        # records for return
        self.targeting_codewords = targeting_codewords
        self.clean_images = clean_images
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(clean_images.shape[0])
        self.converge_iters = np.inf * np.ones(self.batch_size)
        self.best_iters = np.zeros(clean_images.shape[0])
        self.best_consts = np.zeros(clean_images.shape[0])
        self.best_l2s = np.inf * np.ones(clean_images.shape[0])
        self.best_linfs = np.inf * np.ones(clean_images.shape[0])

        # raw_target_logits = self._logits_func([batch_target_imgs])[0]
        raw_target_logits = np.concatenate([self._logits_func([one_target_imgs])[0] for one_target_imgs in batch_target_imgs], axis=0)
        # of shape (batch_size * target_img_pool_size =50 * batch size, len(codeword)

        # compute mean of target logits every pool
        target_logits = []
        for pool_id in range(int(np.ceil(raw_target_logits.shape[0] / tar_pool_size))):
            try:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: (pool_id + 1) * tar_pool_size], axis=0)
                assert len(ave_for_one) == raw_target_logits.shape[1]
            except IndexError:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: ], axis=0)
            target_logits.append(ave_for_one)
        assert len(target_logits) == self.batch_size
        target_logits = np.array(target_logits)
        assert target_logits.shape == (self.batch_size, self.logits_length)

        perturbeds = clean_images.copy()
        previous_losses = [np.zeros(self.batch_size)]*9
        for i in range(max_iter):
            LOTS_grads = self._LOTS_grad_func([perturbeds, target_logits])[0] # of shape (batch size, image shape)
            LOTS_losses = self._LOTS_loss_func([perturbeds, target_logits])[0] # of shape (batch size, image shape)
            previous_losses.append(LOTS_losses)
            previous_losses.pop(0)
            # converge judge
            if i>=10 :
                mean_prevous_losses = np.mean(previous_losses, axis=1)
                for idx, (mpl, ll) in enumerate(zip(mean_prevous_losses, LOTS_losses)):
                    if ll - mpl <= mpl/100:
                        # converged
                        self.converge_iters[idx] = min(i, self.converge_iters[idx])

            # per image-grad normalize
            for idx, grad in enumerate(LOTS_grads):


                grad = grad / np.max(np.abs(grad)) / 255

                perturbeds[idx] = np.clip(perturbeds[idx] - grad, 0, 1)

            # check batch
            is_advs, best_advs, best_distances, best_iters = new_perturbeds(perturbeds, i)
            if np.all(is_advs): # see if improvements
                break
        return self.best_advs, self.best_distances, self.best_iters, self.best_linfs, self.converge_iters


class LogitsLOTSAttack_onehot(KerasModelAttack):
    """
    The original LOTS, excatly the same as the LOTS paper
    """
    def __init__(self, model, hadamard_matrix):
        super(LogitsLOTSAttack_onehot, self).__init__(model)
        keras_model = self.model
        # get logits before activation (softmax)
        logits = keras_model.output.op.inputs[0]
        self.hadamard_matrix = hadamard_matrix
        self.logits_length  = logits.shape[1]
        target_logits = K.placeholder(shape=(None,logits.shape[1]))
        # LOTS_object = K.dot(target_logits - logits_tensor, K.transpose(target_logits - logits_tensor)) / 2
        LOTS_object = K.sum(K.square(target_logits - logits),axis=1) / 2
        LOTS_grad = K.gradients(LOTS_object, keras_model.input)

        self._LOTS_loss_func = K.function([keras_model.input, target_logits], [LOTS_object])
        self._logits_func = K.function([keras_model.input], [logits])
        self._LOTS_grad_func = K.function([keras_model.input, target_logits], LOTS_grad)


    def run_attack(self, clean_images, batch_target_imgs, targeting_codewords, tar_pool_size = 50, **kwargs):
        '''

        :param clean_images: of dim 4 (batch_size, 64, 64, 3)
        :param batch_target_imgs: of dim 5 (batch_size, target_img_pool_size =50, 64, 64, 3)
        :param targeting_codewords: of dim 2 (batch size, len(codeword)
        :param kwargs: max iters and step size
        :return:
        '''

        def new_perturbeds(perturbed_images, current_iter):
            current_logits = self._logits_func([perturbed_images])[0]
            current_labels = np.argmax(softmax(current_logits, axis=1), axis=1)

            # TODO: calculate the target class according to the targeting_codewords
            target_labels = np.argmax(np.matmul(self.targeting_codewords, self.hadamard_matrix), axis=1)
            is_advs = (current_labels == target_labels)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))

                if is_advs[index] and psnr > self.best_distances[index]:
                    self.best_distances[index] = psnr
                    self.best_advs[index] = perturbed
                    self.best_iters[index] = current_iter
                    self.best_l2s[index] = l2
                    self.best_linfs[index] = linf

            return is_advs, self.best_advs, self.best_distances, self.best_iters



        max_iter = kwargs['max_iteration']

        self.batch_size = clean_images.shape[0]

        # records for return
        self.targeting_codewords = targeting_codewords
        self.clean_images = clean_images
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(clean_images.shape[0])
        self.converge_iters = np.inf * np.ones(self.batch_size)
        self.best_iters = np.zeros(clean_images.shape[0])
        self.best_consts = np.zeros(clean_images.shape[0])
        self.best_l2s = np.inf * np.ones(clean_images.shape[0])
        self.best_linfs = np.inf * np.ones(clean_images.shape[0])

        # raw_target_logits = self._logits_func([batch_target_imgs])[0]
        raw_target_logits = np.concatenate([self._logits_func([one_target_imgs])[0] for one_target_imgs in batch_target_imgs], axis=0)
        # of shape (batch_size * target_img_pool_size =50 * batch size, len(codeword)

        # compute mean of target logits every pool
        target_logits = []
        for pool_id in range(int(np.ceil(raw_target_logits.shape[0] / tar_pool_size))):
            try:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: (pool_id + 1) * tar_pool_size], axis=0)
                assert len(ave_for_one) == raw_target_logits.shape[1]
            except IndexError:
                ave_for_one = np.mean(raw_target_logits[pool_id * tar_pool_size: ], axis=0)
            target_logits.append(ave_for_one)
        assert len(target_logits) == self.batch_size
        target_logits = np.array(target_logits)
        assert target_logits.shape == (self.batch_size, self.logits_length)

        perturbeds = clean_images.copy()
        previous_losses = [np.zeros(self.batch_size)]*9
        for i in range(max_iter):
            LOTS_grads = self._LOTS_grad_func([perturbeds, target_logits])[0] # of shape (batch size, image shape)
            LOTS_losses = self._LOTS_loss_func([perturbeds, target_logits])[0] # of shape (batch size, image shape)
            previous_losses.append(LOTS_losses)
            previous_losses.pop(0)
            # converge judge
            if i>=10 :
                mean_prevous_losses = np.mean(previous_losses, axis=1)
                for idx, (mpl, ll) in enumerate(zip(mean_prevous_losses, LOTS_losses)):
                    if ll - mpl <= mpl/100:
                        # converged
                        self.converge_iters[idx] = min(i, self.converge_iters[idx])

            # per image-grad normalize
            for idx, grad in enumerate(LOTS_grads):


                grad = grad / np.max(np.abs(grad)) / 255

                perturbeds[idx] = np.clip(perturbeds[idx] - grad, 0, 1)

            # check batch
            is_advs, best_advs, best_distances, best_iters = new_perturbeds(perturbeds, i)
            if np.all(is_advs): # see if improvements
                break
        return self.best_advs, self.best_distances, self.best_iters, self.best_linfs, self.converge_iters


class SummationLogitsAttack(ECOCAttack):
    def __init__(self, ecoc_model):
        super(SummationLogitsAttack, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model

        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self.codeword_length = len(logits)
        logits_tensor = K.concatenate(logits) # should be of shape (None, code_len)

        target_codewords = K.placeholder(shape=(None, self.codeword_length))

        # want to minimize
        sumloss =  -tf.reduce_sum(logits_tensor * target_codewords, axis=1, name='sum_loss')  # should be of shape (None,)


        sumloss_grad = K.gradients(sumloss, keras_model.input)

        self._logits_fn = K.function([keras_model.input], [logits_tensor])
        self._sumloss_fn = K.function([keras_model.input, target_codewords],[sumloss])
        self._sumloss_grad_fn = K.function([keras_model.input, target_codewords], sumloss_grad)

    def run_attack(self, clean_images, targeting_codewords, step_size, max_iteration):
        '''

        :param clean_images:
        :param targeting_codewords: in {-1,1}
        :param step_size:
        :param max_iteration:
        :return:
        '''
        self.batch_size = clean_images.shape[0]
        self.clean_images = clean_images
        self.targeting_codewords = targeting_codewords
        self.step_size = step_size
        self.max_iteration = max_iteration
        min_, max_ = (0, 1)

        #records for return
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(self.batch_size)
        self.best_iters = np.zeros(self.batch_size)
        self.best_l2s = np.inf * np.ones(self.batch_size)
        self.best_linfs = np.inf * np.ones(self.batch_size)

        def new_perturbeds(perturbed_images, current_iter):
            current_logits = self._logits_fn([perturbed_images])[0]
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            current_codewords = self.model.hadamard_matrix[current_labels]

            is_advs = np.prod(current_codewords == self.targeting_codewords, axis=1)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))

                if is_advs[index] and psnr > self.best_distances[index]:
                    self.best_distances[index] = psnr
                    self.best_advs[index] = perturbed
                    self.best_iters[index] = current_iter
                    self.best_l2s[index] = l2
                    self.best_linfs[index] = linf

            return is_advs, self.best_advs, self.best_distances, self.best_iters

        perturbeds = clean_images.copy()
        sumloss_at_previous_check = np.inf
        for i in range(max_iteration):
            sum_losses = self._sumloss_fn([perturbeds, targeting_codewords])[0]
            sum_loss_grads = self._sumloss_grad_fn([perturbeds, targeting_codewords])[0]


            # normalize grads per image
            for idx, grad in enumerate(sum_loss_grads):
                grad = step_size * grad / np.sqrt(np.sum(np.square(grad)))
                perturbeds[idx] = np.clip(perturbeds[idx] - grad, 0,1)

            is_advs, best_advs, best_distances, best_iters = new_perturbeds(perturbeds, i)

            # early stop checks
            # if np.all(is_advs):
            #     break
            if i % (np.ceil(max_iteration / 10)) == 0:
                # every 10% iterations check loss
                if np.all(sum_losses >= 0.9999 * sumloss_at_previous_check):
                    break  # stop Adam if all in batch has no progress
                sumloss_at_previous_check = sum_losses

        return self.best_advs, self.best_distances, self.best_iters, self.best_l2s, self.best_linfs

class SummationLogitsAttackv2(ECOCAttack):
    def __init__(self, ecoc_model):
        super(SummationLogitsAttackv2, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model

        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self.codeword_length = len(logits)
        logits_tensor = K.concatenate(logits) # should be of shape (None, code_len)

        target_codewords = K.placeholder(shape=(None, self.codeword_length))

        # want to minimize
        sumloss =  -tf.reduce_sum(logits_tensor * target_codewords, axis=1, name='sum_loss')  # should be of shape (None,)


        sumloss_grad = K.gradients(sumloss, keras_model.input)

        self._logits_fn = K.function([keras_model.input], [logits_tensor])
        self._sumloss_fn = K.function([keras_model.input, target_codewords],[sumloss])
        self._sumloss_grad_fn = K.function([keras_model.input, target_codewords], sumloss_grad)

    def run_attack(self, clean_images, targeting_codewords, step_size, max_iteration,
                   initial_constant, num_binary_search):
        '''

        :param clean_images:
        :param targeting_codewords: in {-1,1}
        :param step_size:
        :param max_iteration:
        :return:
        '''
        self.batch_size = clean_images.shape[0]
        self.clean_images = clean_images
        self.targeting_codewords = targeting_codewords
        self.step_size = step_size
        self.max_iteration = max_iteration
        min_, max_ = (0, 1)
        def overall_loss_grad(perturbeds, clean_imgs, targeting_codewords, constants):
            """

            :param perturbeds:
            :param clean_imgs:
            :param who_minus_whom: # e.g. [ 0, 0, 1, -1, 0, 0]  indicates the third minus the forth
            :param constants:
            :param confidence:
            :return:
            """
            # imgs should be all in model space, therefore the loss is in model space, need be convert to attack space when doing optimize
            s = max_ - min_  # should be _max - _min
            # clean_img is for distance penalty
            # gradient_vanished = False

            LOTS_loss = self._sumloss_fn([perturbeds, targeting_codewords])[0]
            LOTS_grads = self._sumloss_grad_fn([perturbeds, targeting_codewords])[0]  # of shape (batch size, image shape)

            # L2 distance loss, grad w.r.t. img
            squared_l2_distances = np.sum((perturbeds - clean_imgs) ** 2, axis=(1, 2, 3)) / s ** 2
            squared_l2_distance_grads = (2 / s ** 2) * (perturbeds - clean_imgs)

            # total loss, grad w.r.t. img
            overall_losses = squared_l2_distances + constants * LOTS_loss
            overall_grads = squared_l2_distance_grads + np.array([c * cg for c,cg in zip(constants, LOTS_grads)])
            return overall_losses, overall_grads

        #records for return
        self.best_advs = [None] * self.batch_size
        self.best_distances = -np.inf * np.ones(self.batch_size)
        self.best_iters = np.zeros(self.batch_size)
        self.best_l2s = np.inf * np.ones(self.batch_size)
        self.best_linfs = np.inf * np.ones(self.batch_size)
        self.best_bs_nums = np.zeros(self.batch_size)
        self.best_consts = np.zeros(self.batch_size)

        def new_perturbeds(perturbed_images, current_iter, current_constants, current_bs_num):
            current_logits = self._logits_fn([perturbed_images])[0]
            current_labels = np.argmax(np.matmul(current_logits, self.model.hadamard_matrix.T), axis=1)
            current_codewords = self.model.hadamard_matrix[current_labels]

            is_advs = np.prod(current_codewords == self.targeting_codewords, axis=1)
            for index, (perturbed, clean_img) in enumerate(zip(perturbed_images, self.clean_images)):
                psnr = compute_psnr(perturbed, clean_img)
                l2 = np.sqrt(np.sum(np.square(perturbed - clean_img)))
                linf = np.max(np.abs(perturbed - clean_img))

                if is_advs[index] and psnr > self.best_distances[index]:
                    self.best_distances[index] = psnr
                    self.best_advs[index] = perturbed
                    self.best_iters[index] = current_iter
                    self.best_l2s[index] = l2
                    self.best_linfs[index] = linf
                    self.best_bs_nums[index] = current_bs_num
                    self.best_consts[index] = current_constants[index]

            return is_advs, self.best_advs, self.best_distances, self.best_iters, self.best_consts, self.best_bs_nums

        perturbeds = clean_images.copy()
        consts = initial_constant * np.ones(self.batch_size)
        lower_bounds = np.zeros(self.batch_size)
        upper_bound = np.inf * np.ones(self.batch_size)
        for binary_search_step in range(num_binary_search):
            found_advs = [False] * self.batch_size  # found adv with the current const
            sumloss_at_previous_check = np.inf
            for i in range(max_iteration):
                sum_losses, sum_loss_grads = overall_loss_grad(perturbeds, clean_images, targeting_codewords, consts)

                # normalize grads per image
                for idx, grad in enumerate(sum_loss_grads):
                    grad = step_size * grad / np.sqrt(np.sum(np.square(grad)))
                    perturbeds[idx] = np.clip(perturbeds[idx] - grad, 0,1)

                is_advs, best_advs, best_distances, best_iters, best_consts, best_bs_nums\
                    = new_perturbeds(perturbeds, i, consts, binary_search_step)
                found_advs = np.logical_or(found_advs, is_advs)

                if i % (np.ceil(max_iteration / 10)) == 0:
                    # every 10% iterations check loss
                    if np.all(sum_losses >= 0.9999 * sumloss_at_previous_check):
                        break  # stop Adam if all in batch has no progress
                    sumloss_at_previous_check = sum_losses
            # binary search part
            for idx in range(self.batch_size):
                if found_advs[idx]:
                    # success
                    upper_bound[idx] = consts[idx]
                else:
                    # fails
                    lower_bounds[idx] = consts[idx]

                if upper_bound[idx] == np.inf:
                    # exponential search
                    consts[idx] *= 10
                else:
                    # binary search
                    consts[idx] = (lower_bounds[idx] + upper_bound[idx]) / 2

        return self.best_advs, self.best_distances, self.best_iters, self.best_l2s, self.best_linfs,\
               self.best_consts, self.best_bs_nums

class JacobianLogitsMDis(ECOCAttack):
    """ not finished """
    def __init__(self, ecoc_model):
        super(JacobianLogitsMDis, self).__init__(ecoc_model)
        keras_model = ecoc_model.keras_model
        # get logits before activation (tanh)
        individual_activations = keras_model.layers[-1].input
        logits = [activate.op.inputs[0] for activate in individual_activations]
        self._logits_func = K.function([keras_model.input], logits)
        # new "Jacobian at logits level"
        Jacbs_logits_list = [K.gradients(li, keras_model.input) for li in logits]
        Jacbs_logits = tf.stack(Jacbs_logits_list, name='Jacb_logits')
        self._Jacbian_logits_fn = K.function([keras_model.input],[Jacbs_logits])













# if __name__ == '__main__':
    # # for debugging
    # from glob import glob
    #
    # bit_model_weights_list = glob(
    #     r'D:\LableCodingNetwork\trained-models\gtsrb\ECOC\Hadamard32_surrogate_weights_freeze6_bit_*\final_trained_weights.hdf5')
    # bit_model_weights_list.sort(key=lambda x: int(x.split('bit_')[-1].split('\\')[0]))
    # HADAMARD_MATRIX = np.load(r'D:\LableCodingNetwork\ecoc_hadamard_gtsrb\hadamard32.npy')
    # TOP32GTSRB_CATEGORIES = np.load(r'D:\LableCodingNetwork\database\gtsrb\gysrb_top32category_label.npy')
    # ecoc_model = ECOC_Hadamard_Model(bit_model_weights_list, HADAMARD_MATRIX)
    # from utils.readTrafficSigns import Hadamard_labels
    #
    # TEST_IMG_FOLDER = r'D:\LableCodingNetwork\database\gtsrb\Final_Test\Images'
    # bitwise_ground_labels_list = []
    # for i in range(len(bit_model_weights_list)):
    #     BIT_MODEL_INDEX = i + 1
    #     test_img_paths, test_bit_model_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES, HADAMARD_MATRIX,
    #                                                             BIT_MODEL_INDEX)
    #     bitwise_ground_labels_list.append(test_bit_model_labels)
    # bitwise_ground_label_matrix = np.array(bitwise_ground_labels_list)
    # test_img_paths, ground_index_label_accodring_gtsrb_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES,
    #                                                                             HADAMARD_MATRIX,
    #                                                                             0)
    # # prepare img all together
    # from keras.preprocessing.image import load_img, img_to_array
    #
    # test_imgs = np.array([img_to_array(load_img(path, target_size=(64, 64))) / 255 for path in test_img_paths])
    # to_be_attack_imgs_index = np.load(
    #     r'D:\LableCodingNetwork\trained-models\gtsrb\ECOC\Hadamard32_surrogate_weights_freeze6_100_tobe_attack_img_index_on_Hadamard_labels_output.npy')
    # to_be_attack_imgs = test_imgs[to_be_attack_imgs_index]
    # to_be_attack_imgs_bit_label_matrix = bitwise_ground_label_matrix.T[to_be_attack_imgs_index]  # shape = (100,32)
    #
    # # attack
    # Attack = ECOCAttack(ecoc_model)
    # error = np.ones(32)
    # error[np.random.choice(32, 5, replace=False)] = -1
    # error[0] = 1
    # adv = Attack.CWBattack(to_be_attack_imgs[0][np.newaxis],
    #                        initial_constant=1,
    #                        targeting_un_decoded=to_be_attack_imgs_bit_label_matrix[0] * error)

    # '''test new jac attack'''
    # adv_new_jac = Attack.NewJacAttack(to_be_attack_imgs[0][np.newaxis],HADAMARD_MATRIX[2])