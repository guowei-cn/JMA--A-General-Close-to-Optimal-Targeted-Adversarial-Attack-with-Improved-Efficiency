# import pudb; pu.db
import argparse
import sys
from glob import glob
import numpy as np

from ECOC.gtsrb.models import ECOC_Hadamard_Model
from utils.readTrafficSigns import Hadamard_labels
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from core_attack.Attacks import JacbMDisTryAndOptimize
import os, time
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.random.seed(19930117)
a = np.random.get_state()
np.random.set_state(a)

HADAMARD_MATRIX = np.load('ECOC/gtsrb/hadamard32.npy')
TRAINED_MODEL_PATH = 'Model/gtsrb/model_weight/Hadamard32_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5'
TEST_IMG_FOLDER = 'Dataset/gtsrb/Final_Test/Images'
TOP32GTSRB_CATEGORIES = np.load('Dataset/gtsrb/gtsrb_top32category_label.npy')

index_path = 'Dataset/gtsrb/to_be_attacked_200_gtsrb_testset_index_June16.npy'
error_path = 'Dataset/gtsrb/to_be_attacked_200_gtsrb_testset_error_patterns_June16.npy'

bit_model_weights_list = glob(TRAINED_MODEL_PATH)
bit_model_weights_list.sort(key=lambda x: int(x.replace("\\", "/").split('bit_')[-1].split('/')[0]))


# get (num_example, 32) bit-wise ground truth label/codeword
bitwise_ground_labels_list = []
for i in range(len(bit_model_weights_list)):
    BIT_MODEL_INDEX = i + 1
    test_img_paths, test_bit_model_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES, HADAMARD_MATRIX,
                                                            BIT_MODEL_INDEX)
    bitwise_ground_labels_list.append(test_bit_model_labels)

bitwise_ground_label_matrix = np.array(bitwise_ground_labels_list)
test_img_paths, ground_index_label_accodring_gtsrb_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES,
                                                                            HADAMARD_MATRIX, 0)

# to be attacked imgs

# to_be_attack_imgs_index = np.load(
#     os.path.join(MODEL_ROOT, 'gtsrb/ECOC/to_be_attacked_200_gtsrb_testset_index_June16.npy')) # new
to_be_attack_imgs_index = np.load(index_path) # very old ones for EL submission

to_be_attack_imgs_bit_label_matrix = bitwise_ground_label_matrix.T[to_be_attack_imgs_index]  # shape = (100,32)
to_be_attack_imgs_paths = test_img_paths[to_be_attack_imgs_index]
# error_patterns = do_you_have_error_pattern('/media/hdddati2/bzhang/trained_models/gtsrb/ECOC/to_be_attacked_200_gtsrb_testset_error_patterns_June16.npy')
error_patterns = np.load(error_path)  # very old ones for EL submission

# initialize model and attack
ecoc_model = ECOC_Hadamard_Model(bit_model_weights_list, HADAMARD_MATRIX)
Attack = JacbMDisTryAndOptimize(ecoc_model)


def JacbMDisTryAndOptimize_ecoc_gtsrb(record_save_file, **kwargs):
    save_folder = os.path.dirname(record_save_file)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # loop imgs to attack
    success = []
    distances_psnr_suc, distances_psnr_fail = [], []
    iterations_suc, iterations_fail = [], []
    time_costs_suc, time_costs_fail = [], []

    for idx, (img_path, ground_codeword, error) in enumerate(zip(to_be_attack_imgs_paths, to_be_attack_imgs_bit_label_matrix, error_patterns)):
        if idx > 200:
            break
        clean_img = img_to_array(load_img(img_path, target_size=(64, 64)))[np.newaxis] / 255
        assert np.prod(ecoc_model.predict(clean_img, output_type='codeword') == ground_codeword)
        # error pattern indicates as -1(error) and +1(clear)
        raw_code = error * ground_codeword
        clean_class = np.argmax(np.matmul(ground_codeword, ecoc_model.hadamard_matrix.T), axis=-1)
        target_class = np.argmax(np.matmul(raw_code, ecoc_model.hadamard_matrix.T), axis=-1)
        target_codeword = ecoc_model.hadamard_matrix[target_class]

        with open(record_save_file, 'a') as f:
            f.write('Attacking number {} img {}\n'
              'clean_codeword is {}, clean_class is {}\n'
              'targeting decoded codeword is {}, targeting class is {}\n'
              .format(idx, img_path,
                      ground_codeword, clean_class,
                      target_codeword, target_class))

        # print(f"Starting attack on image #{idx} - {os.path.basename(img_path)}")
        _start = time.perf_counter()
        adv, psnr, iteration = Attack.run_attack(clean_img, targeting_codeword=target_codeword, **kwargs)
        time_cost = time.perf_counter() - _start

        flag = adv is not None
        success.append(flag)
        # print(f"Attack ended for image #{idx} after {time_cost} seconds (success={flag})")
        #
        # print('Success = {}, psnr = {}\n num_iter = {}, cost {} seconds\n\n'.format(flag, psnr, iteration, time_cost))
        with open(record_save_file, 'a') as f:
            f.write('Success = {}, psnr = {}\n num_iter = {}, cost {} seconds\n\n'.format(flag, psnr, iteration, time_cost))
        if flag:
            time_costs_suc.append(time_cost)
            distances_psnr_suc.append(psnr)
            iterations_suc.append(iteration)
            np.save(os.path.join(save_folder, os.path.basename(img_path).replace('.ppm', '.npy')),
                                  np.squeeze(adv))
        else:
            time_costs_fail.append(time_cost)
            distances_psnr_fail.append(psnr)
            iterations_fail.append(iteration)

    # initial statis analysis of J_ransss

    asr = success.count(True) / len(success)
    ave_psnr_suc = np.mean(distances_psnr_suc)
    ave_iteration_suc = np.mean(iterations_suc)
    ave_time_consumption_suc = np.mean(time_costs_suc)
    ave_psnr_fail = np.mean(distances_psnr_fail)
    ave_iteration_fail = np.mean(iterations_fail)
    ave_time_consumption_fail = np.mean(time_costs_fail)

    print('ASR = {}, ave PSNR suc/fail = {:.2f}/{:.2f}, ave num_iter suc/fail = {:.2f}/{:.2f}, ave time consumption suc/fail = {:.2f}/{:.2f}'.format(asr, ave_psnr_suc, ave_psnr_fail, ave_iteration_suc, ave_iteration_fail, ave_time_consumption_suc, ave_time_consumption_fail))
    with open(record_save_file, 'a') as f:
        f.write('ASR = {}, ave PSNR suc/fail = {:.2f}/{:.2f}, ave num_iter suc/fail = {:.2f}/{:.2f}, ave time consumption suc/fail = {:.2f}/{:.2f}'.format(asr,
                                                                                                                                                           ave_psnr_suc,
                                                                                                                                                           ave_psnr_fail,
                                                                                                                                                           ave_iteration_suc,
                                                                                                                                                           ave_iteration_fail,
                                                                                                                                                           ave_time_consumption_suc,
                                                                                                                                                           ave_time_consumption_fail))
    print(kwargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', help='model path', default='..\MultiLabel\model4voc2012.h5')
    parser.add_argument('-s', help='step size', type=float, default=0.5)
    parser.add_argument("-i", help="max iteration", default=200,
                        type=int)

    parser.add_argument('-c', help='confidence', type=int, default=0)
    args = parser.parse_args()

    confidence, step_size, max_iterations = args.c, args.s, args.i
    kwargs = {
        'step_size': step_size,
        'max_iteration': max_iterations,
        'confidence': confidence
    }

    path = \
        os.path.join(os.getcwd(),
                     'ECOC/gtsrb/eps_{}_maxiter_{}/record.txt'.format(kwargs['step_size'], kwargs['max_iteration']))
    JacbMDisTryAndOptimize_ecoc_gtsrb(path, **kwargs)