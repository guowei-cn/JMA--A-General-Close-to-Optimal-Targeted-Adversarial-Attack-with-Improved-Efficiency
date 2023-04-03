"""object that store things need to be record by adversarial"""
"""
used in experiment with Bowen attack and ECOC model, not useful for Jacobian attack
UltimateAdversarial class is the final used one
"""
from utils.math_func import compute_psnr
import numpy as np


class LatentAdversarial:
    def __init__(self, clean_img, is_target_attack=True, raw_target_codeword = None, latent_model=None, clean_img_path=None,
                 better_criterion='psnr', decode = False):
        self.__decode = decode
        self.__clean_img = clean_img
        self.__model = latent_model
        self.__is_target_attack = is_target_attack
        self.__target_codeword = raw_target_codeword
        self.__clean_img_path = clean_img_path
        assert better_criterion in ['psnr', 'linf']
        self.__better_criterion = better_criterion
        self.__best_distance = np.inf  # may need to change this accodring to better_criterion that comes latter
        self.__best_adv = None # may need to change according to attacks that come later
        self.__clean_codeword = latent_model.predict(clean_img, output_type='undecoded')
        # self.__target_error = np.mod(self.__clean_codeword - self.__target_codeword, 2)
        self.__target_error = 1*np.array(self.__clean_codeword != self.__target_codeword)
        self._other_records = None

        if decode:
            assert np.all(self.__clean_codeword == latent_model._decoding(self.__clean_codeword))
            self.__clean_codeword = latent_model._decoding(self.__clean_codeword)
            self.__target_codeword = latent_model._decoding(raw_target_codeword)
            self.__target_class = latent_model._codeword2label(self.__target_codeword)

            # target_codeword could be illegal, while latent_model doesn't output undecoded codeword yet, so decode target_codeword here


    def new_perturbed(self, perturbed, **kwargs):
        # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
        output_type = 'codewords' if self.__decode else 'undecoded'
        if not self.__is_target_attack:
            raise TypeError('Non-target attack not implement yet')
        predict_codeword = self.__model.predict(perturbed, output_type=output_type)
        # hamming_dis = np.sum(np.mod(predict_codeword + ori_codeword, 2))
        is_adv = (self.__target_codeword == predict_codeword).all()
        linf = np.max(np.abs(self.__clean_img - perturbed))
        psnr = -compute_psnr(self.__clean_img, perturbed) # set it to minus so aligned with other distance that smaller is better
        if self.__better_criterion == 'psnr':
            new_distance = psnr
        elif self.__better_criterion == 'linf':
            new_distance = linf
        else:
            new_distance = np.inf
        # pick smallest distance
        if is_adv and self.__best_distance > new_distance:
            self.__best_distance = new_distance
            self.__best_adv = perturbed
            self.update_records(**kwargs)

        return is_adv, linf, psnr

    def update_records(self, **kwargs):
        self._other_records = kwargs

    @property
    def other_records(self):
        return self._other_records

    @property
    def model(self):
        return self.__model

    @property
    def is_decoded(self):
        return self.__decode

    @property
    def best_adv(self):
        return self.__best_adv

    @property
    def better_adv_criterion(self):
        return self.__better_criterion

    @property
    def clean_img(self):
        return self.__clean_img

    @property
    def clean_codeword(self):
        return self.__clean_codeword

    @property
    def target_codeword(self):
        return self.__target_codeword

    @property
    def target_error(self):
        return self.__target_error

    @property
    def best_distance(self):
        return self.__best_distance


class BetterAdversarial:

    def __init__(self, clean_img, latent_model=None, is_target_attack=True, target_output = None, clean_img_path=None,
                 better_criterion='psnr', output_type = False):
        self.__output_type = output_type
        self.__clean_img = clean_img
        self.__model = latent_model
        self.__is_target_attack = is_target_attack
        self.__target_output = target_output
        self.__clean_img_path = clean_img_path
        assert better_criterion in ['psnr', 'linf']
        self.__better_criterion = better_criterion
        self.__best_distance = np.inf  # may need to change this accodring to better_criterion that comes latter
        self.__best_adv = None # may need to change according to attacks that come later
        self.__clean_output = latent_model.predict(clean_img, output_type=output_type)
        # self.__target_error = np.mod(self.__clean_codeword - self.__target_codeword, 2)
        self.__target_error = 1*np.array(self.__clean_output != self.__target_output)
        self._other_records = {}


    def new_perturbed(self, perturbed, **kwargs):
        # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
        output_type = self.__output_type
        if not self.__is_target_attack:
            raise TypeError('Non-target attack not implement yet')
        predict_output = self.__model.predict(perturbed, output_type=output_type)

        is_adv = (self.__target_output == predict_output).all()
        linf = np.max(np.abs(self.__clean_img - perturbed))
        psnr = -compute_psnr(self.__clean_img, perturbed) # set it to minus so aligned with other distance that smaller is better
        if self.__better_criterion == 'psnr':
            new_distance = psnr
        elif self.__better_criterion == 'linf':
            new_distance = linf
        else:
            new_distance = np.inf
        # pick smallest distance
        if is_adv and self.__best_distance > new_distance:
            self.__best_distance = new_distance
            self.__best_adv = perturbed
            self.update_records(**kwargs)

        return is_adv, linf, psnr

    def update_records(self, **kwargs):
        self._other_records.update(kwargs)

    @property
    def other_records(self):
        return self._other_records

    @property
    def model(self):
        return self.__model

    @property
    def is_decoded(self):
        return self.__output_type

    @property
    def best_adv(self):
        return self.__best_adv

    @property
    def better_adv_criterion(self):
        return self.__better_criterion

    @property
    def clean_img(self):
        return self.__clean_img

    @property
    def clean_output(self):
        return self.__clean_output

    @property
    def target_codeword(self):
        return self.__target_output

    @property
    def target_error(self):
        return self.__target_error

    @property
    def best_distance(self):
        return self.__best_distance


class UltimateAdversarial(object):

    def __init__(self, clean_img, latent_model=None, is_target_attack=True, target_output = None, clean_img_path=None,
                 better_criterion='psnr', output_type = False, conf_thres = 0.0, strict_conf_thres = True):
        self.__output_type = output_type
        self.__clean_img = clean_img
        self.__model = latent_model
        self.__is_target_attack = is_target_attack
        self.__target_output = target_output
        self.__clean_img_path = clean_img_path
        assert better_criterion in ['psnr', 'linf']
        self.__better_criterion = better_criterion
        self.__best_distance = np.inf if better_criterion == 'linf' else -np.inf # may need to change this accodring to better_criterion that comes latter
        self.__best_adv = None # may need to change according to attacks that come later
        self.__clean_output = latent_model.predict(clean_img, output_type=output_type)
        # print('------- inside adversarial ------\n'
        #       '---___init___\n'
        #       'self.__clean_output = latent_model.predict(clean_img, output_type=output_type) = {}, output_type={}'.format(self.clean_output, output_type))
        # self.__target_error = np.mod(self.__clean_codeword - self.__target_codeword, 2)
        self.__target_error = 1*np.array(self.__clean_output != self.__target_output)
        self.__conf_thres = conf_thres
        self.__strict_conf_thres = strict_conf_thres
        self._other_records = {}

    def is_non_target_adv(self, prediction):
        # print('prediction={} self.__clean_output={}'.format(prediction, self.__clean_output))
        # print('not np.array_equal(prediction, self.__clean_output)', not np.array_equal(prediction, self.__clean_output))
        return not np.array_equal(prediction, self.__clean_output)
    def is_target_adv(self, prediction):
        prediction = np.squeeze(prediction)
        # print('np.array_equal(prediction, self.__target_output)', np.array_equal(prediction, self.__target_output))
        # print('prediction={} self.__target_output={}'.format(prediction, self.__target_output))
        return np.array_equal(prediction, self.__target_output)


    def new_perturbed(self, perturbed, output_type=None, **kwargs):
        # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
        if output_type is None:
            output_type = self.__output_type

        is_adv_fn = self.is_target_adv if self.__is_target_attack else self.is_non_target_adv
        predict_output = self.__model.predict(perturbed, output_type=output_type)
        # print('self.__model.predict(perturbed, output_type=output_type)', predict_output)

        is_adv = is_adv_fn(predict_output)
        if self.__strict_conf_thres:
            # not is_adv anymore if conf not satistified
            try:
                conf = kwargs['conf']
            except KeyError:
                print('No key "conf" value been feed to new_perturbed method, no strict thres applied')
                conf = np.inf
            if conf < self.__conf_thres:
                is_adv = False

        linf = np.max(np.abs(self.__clean_img - perturbed))
        psnr = compute_psnr(self.__clean_img, perturbed) # set it to minus so aligned with other distance that smaller is better
        if self.__better_criterion == 'psnr':
            new_distance = psnr
            if is_adv and self.__best_distance < new_distance:
                self.__best_distance = new_distance
                self.__best_adv = perturbed
                self.update_records(**kwargs)
        elif self.__better_criterion == 'linf':
            new_distance = linf
            if is_adv and self.__best_distance > new_distance:
                self.__best_distance = new_distance
                self.__best_adv = perturbed
                self.update_records(**kwargs)

        return is_adv, linf, psnr

    def update_records(self, **kwargs):
        self._other_records.update(kwargs)

    @property
    def other_records(self):
        return self._other_records

    @property
    def model(self):
        return self.__model

    @property
    def is_decoded(self):
        return self.__output_type

    @property
    def best_adv(self):
        return self.__best_adv

    @property
    def better_adv_criterion(self):
        return self.__better_criterion

    @property
    def clean_img(self):
        return self.__clean_img

    @property
    def clean_output(self):
        return self.__clean_output

    @property
    def target_output(self):
        return self.__target_output

    @property
    def target_error(self):
        return self.__target_error

    @property
    def best_distance(self):
        return self.__best_distance


class UltimateAdversarial_multilabel(object):

    def __init__(self, clean_img, latent_model=None, is_target_attack=True, target_output = None, clean_img_path=None,
                 better_criterion='psnr', conf_thres = 0.0, strict_conf_thres = True):
        self.__clean_img = clean_img
        self.__model = latent_model
        self.__is_target_attack = is_target_attack
        self.__target_output = target_output
        self.__clean_img_path = clean_img_path
        assert better_criterion in ['psnr', 'linf']
        self.__better_criterion = better_criterion
        self.__best_distance = np.inf if better_criterion == 'linf' else -np.inf # may need to change this accodring to better_criterion that comes latter
        self.__best_adv = None # may need to change according to attacks that come later
        self.__clean_output = latent_model.predict(clean_img)
        self.__best_hamming_loss = np.inf
        # print('------- inside adversarial ------\n'
        #       '---___init___\n'
        #       'self.__clean_output = latent_model.predict(clean_img, output_type=output_type) = {}, output_type={}'.format(self.clean_output, output_type))
        # self.__target_error = np.mod(self.__clean_codeword - self.__target_codeword, 2)
        self.__target_error = 1*np.array(np.sign(self.__clean_output-0.5) != self.__target_output)
        self.__conf_thres = conf_thres
        self.__strict_conf_thres = strict_conf_thres
        self._other_records = {}

    def is_non_target_adv(self, prediction):
        # print('prediction={} self.__clean_output={}'.format(prediction, self.__clean_output))
        # print('not np.array_equal(prediction, self.__clean_output)', not np.array_equal(prediction, self.__clean_output))
        return not np.array_equal(prediction, self.__clean_output)
    def is_target_adv(self, prediction):
        prediction = np.squeeze(prediction)
        # print('np.array_equal(prediction, self.__target_output)', np.array_equal(prediction, self.__target_output))
        # print('prediction={} self.__target_output={}'.format(prediction, self.__target_output))
        return np.array_equal(prediction, self.__target_output)


    def new_perturbed(self, perturbed, output_type=None, **kwargs):
        # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
        # predict_output = self.__model.predict(perturbed)
        # print('self.__model.predict(perturbed, output_type=output_type)', predict_output)

        predict_output = np.sign(np.squeeze(self.model.predict(perturbed)) - 0.5)

        is_adv = np.array_equal(predict_output, self.__target_output)
        if self.__strict_conf_thres:
            # not is_adv anymore if conf not satistified
            try:
                conf = kwargs['conf']
            except KeyError:
                print('No key "conf" value been feed to new_perturbed method, no strict thres applied')
                conf = np.inf
            if conf < self.__conf_thres:
                is_adv = False

        linf = np.max(np.abs(self.__clean_img - perturbed))
        psnr = compute_psnr(self.__clean_img, perturbed) # set it to minus so aligned with other distance that smaller is better
        hamming_loss = np.sum(predict_output != self.__target_output) / self.model.output_shape[1]
        if hamming_loss < self.__best_hamming_loss:
            # and psnr > self.best_psnr:
            self.__best_adv = perturbed
            self.__best_hamming_loss = hamming_loss
            self.__best_distance = psnr
            self.update_records(**kwargs)
        #
        # if self.__better_criterion == 'psnr':
        #     new_distance = psnr
        #     if is_adv and self.__best_distance < new_distance:
        #         self.__best_distance = new_distance
        #         self.__best_adv = perturbed
        #         self.update_records(**kwargs)
        # elif self.__better_criterion == 'linf':
        #     new_distance = linf
        #     if is_adv and self.__best_distance > new_distance:
        #         self.__best_distance = new_distance
        #         self.__best_adv = perturbed
        #         self.update_records(**kwargs)

        return is_adv, linf, psnr

    def update_records(self, **kwargs):
        self._other_records.update(kwargs)

    @property
    def other_records(self):
        return self._other_records

    @property
    def model(self):
        return self.__model


    @property
    def best_adv(self):
        return self.__best_adv

    @property
    def better_adv_criterion(self):
        return self.__better_criterion

    @property
    def clean_img(self):
        return self.__clean_img

    @property
    def clean_output(self):
        return self.__clean_output

    @property
    def target_output(self):
        return self.__target_output

    @property
    def target_error(self):
        return self.__target_error

    @property
    def best_distance(self):
        return self.__best_distance


# class BCHAdversarial:
#     def __init__(self, clean_img, is_target_attack=True, raw_target_codeword = None, bch_model=None, clean_img_path=None,
#                  better_criterion='psnr'):
#         self.__clean_img = clean_img
#         self.__model = bch_model
#         self.__is_target_attack = is_target_attack
#         self.__target_codeword = raw_target_codeword
#         self.__clean_img_path = clean_img_path
#         assert better_criterion in ['psnr', 'linf']
#         self.__better_criterion = better_criterion
#         self.__best_distance = np.inf  # may need to change this accodring to better_criterion that comes latter
#         self.__best_adv = None # may need to change according to attacks that come later
#         self.__clean_codeword = bch_model.predict(clean_img, output_type='codewords')
#         if raw_target_codeword is not None and hasattr(bch_model, 'codeword_to_label'):
#             self.__target_class = bch_model.codeword_to_label(bch_model.decode(raw_target_codeword)[:5])
#             self.__target_error = np.mod(self.__clean_codeword - raw_target_codeword, 2)
#             # target_codeword could be illegal, while latent_model doesn't output undecoded codeword yet, so decode target_codeword here
#             self.__target_codeword = bch_model.decode(raw_target_codeword)
#     def new_perturbed(self, perturbed):
#         # when a new perturbed was generated, use this to see if it is better and whether records needs to be updated
#         if not self.__is_target_attack:
#             raise TypeError('Non-target attack not implement yet')
#         predict_codeword = self.__model.predict(perturbed, output_type='codewords')
#         # hamming_dis = np.sum(np.mod(predict_codeword + ori_codeword, 2))
#         is_adv = (self.__target_codeword == predict_codeword).all()
#         linf = np.max(np.abs(self.__clean_img - perturbed))
#         psnr = -compute_psnr(self.__clean_img, perturbed) # set it to minus so aligned with other distance that smaller is better
#         if self.__better_criterion == 'psnr':
#             new_distance = psnr
#         elif self.__better_criterion == 'linf':
#             new_distance = linf
#         else:
#             new_distance = np.inf
#         # pick smallest distance
#         if is_adv and self.__best_distance > new_distance:
#             self.__best_distance = new_distance
#             self.__best_adv = perturbed
#         # todo think about other things need record, like eps or something
#         return is_adv, linf, psnr
#
#     @property
#     def best_adv(self):
#         return self.__best_adv
#
#     @property
#     def better_adv_criterion(self):
#         return self.__better_criterion
#
#     @property
#     def clean_img(self):
#         return self.__clean_img
#
#     @property
#     def clean_codeword(self):
#         return self.__clean_codeword
#
#     @property
#     def target_error(self):
#         return self.__target_error
#
#     @property
#     def best_distance(self):
#         return self.__best_distance

