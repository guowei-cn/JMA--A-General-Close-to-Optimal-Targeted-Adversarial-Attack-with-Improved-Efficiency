import argparse
from glob import glob
import numpy as np
import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ECOC.cifar10.data_handler import get_tobe_attacked_imgs
from ECOC.cifar10.models import ECOC_Hadamard_Model
from core_attack.Attacks import JacbMDisTryAndOptimize
import os, time
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(19930117)
a = np.random.get_state()
np.random.set_state(a)
# detect the envir is in server of my own pc

CUDA_VISIBLE_DEVICES="gpu"

# load models & initialize attack
bit_model_weights_paths = glob('Model/cifar10/model_weight/Hadamard16_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5')
bit_model_weights_paths.sort(key=lambda x: int(x.split('bit_')[-1].replace("\\", "/").split('/')[0]))

HADAMARD_MATRIX = np.load('ECOC/cifar10/hadamard16.npy')

# load to-be attacked images
# index_path = os.path.join(MODEL_DIR,'cifar10/ECOC/index_combine_old.npy')
# # error_path = os.path.join(MODEL_DIR,'cifar10/ECOC/error_combine_old.npy')
index_path = 'Dataset/cifar10/to_be_attacked_200_cifar10_testset_June26_index.npy'
error_path = 'Dataset/cifar10/to_be_attacked_200_cifar10_testset_June26_error_pattern.npy'

tobe_attacked_clean_images, ground_cifar_labels, ground_hadamard_codewords = \
    get_tobe_attacked_imgs(hadamard_matrix=HADAMARD_MATRIX, cifar_testset_index=np.load(index_path))
error_patterns = np.load(error_path)

# initialize model and attack
ecoc_model = ECOC_Hadamard_Model(bit_model_weights_paths, HADAMARD_MATRIX[:10,:])
Attack = JacbMDisTryAndOptimize(ecoc_model)

def JacbMDisTryAndOptimize_ecoc_cifar(record_save_file, **kwargs):
    save_folder = os.path.dirname(record_save_file)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # loop imgs to attack
    success = []
    distances_psnr = []
    iterations = []
    time_costs = []


    Jacob_0 = 0
    for idx, (clean_img, ground_codeword, error) in enumerate(zip(tobe_attacked_clean_images, ground_hadamard_codewords, error_patterns)):
        if idx <0:
            continue
        if idx > 200:
            break
        clean_img = clean_img[np.newaxis]
        assert np.prod(ecoc_model.predict(clean_img, output_type='codeword') == ground_codeword)
        # error pattern indicates as -1(error) and +1(clear)
        raw_code = error * ground_codeword
        clean_class = np.argmax(np.matmul(ground_codeword, ecoc_model.hadamard_matrix.T), axis=-1)
        target_class = np.argmax(np.matmul(raw_code, ecoc_model.hadamard_matrix.T), axis=-1)
        if target_class == clean_class:
            target_class = (target_class+1)%10
        target_codeword = ecoc_model.hadamard_matrix[target_class]
        # print('Attacking number {} img in selected set, image L2norm={}\n'
        #       'clean hadamard is {}, clean class is {}\n'
        #       'targeting decoded is {}, targeting class is {}\n'
        #       .format(idx, np.sqrt(np.sum(np.square(clean_img))),
        #               ground_hadamard, clean_class,
        #               target_decoded, target_class))
        with open(record_save_file, 'a') as f:
            f.write('Attacking number {} img in selected set, image L2norm={}\n'
              'clean codeword is {}, clean class is {}\n'
              'targeting decoded is {}, targeting class is {}\n'
              .format(idx, np.sqrt(np.sum(np.square(clean_img))),
                      ground_codeword, clean_class,
                      target_codeword, target_class))

        _start = time.clock()
        adv, psnr, iteration = Attack.run_attack(clean_img, targeting_codeword=target_codeword, **kwargs)
        time_cost = time.clock() - _start
        time_costs.append(time_cost)

        # print('One image consumed time:', time.clock() - _start)
        flag = adv is not None
        success.append(flag)
        # print('Success = {}, psnr = {}\n num_iter = {}, cost {} seconds\n\n'.format(flag, psnr, iteration, time_cost))
        with open(record_save_file, 'a') as f:
            f.write('Success = {}, psnr = {}\n num_iter = {}, cost {} seconds\n\n'.format(flag, psnr, iteration, time_cost))
        if flag:
            distances_psnr.append(psnr)
            iterations.append(iteration)
            np.save(os.path.join(save_folder, 'number{}img_cleanL2{}.npy'
                                 .format(idx, np.sqrt(np.sum(np.square(clean_img))))), np.squeeze(adv))

    # initial statis analysis of J_ransss


    asr = success.count(True) / len(success)
    ave_psnr = np.mean(distances_psnr)
    ave_iteration = np.mean(iterations)
    ave_time_consumption = np.mean(time_costs)

    print('ASR = {}, ave PSNR = {}, ave num_iter = {}, ave time consumption = {}'.format(asr, ave_psnr, ave_iteration, ave_time_consumption))
    with open(record_save_file, 'a') as f:
        f.write('\n\nASR = {}, ave PSNR = {}, ave num_iter = {}, ave time consumption = {}\n'
                'been attacked image index path: {}\n'
                '               error pattern path: {}\n'.format(asr, ave_psnr, ave_iteration, ave_time_consumption,
                                                                 index_path, error_path))
    print(kwargs)
    return asr, ave_psnr, ave_iteration
    # print(np.mean(J_rankss, axis=0))


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
        os.path.join(os.getcwd(), 'ECOC/cifar10/STP={}MI{}Conf{}/record.txt')\
            .format(
            round(kwargs['step_size'],2),
            kwargs['max_iteration'],
            kwargs['confidence'],
        )
    asr, ave_psnr, ave_iteration = JacbMDisTryAndOptimize_ecoc_cifar(path, **kwargs)
    print(path)
