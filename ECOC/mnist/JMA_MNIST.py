import argparse
from glob import glob
import numpy as np
from ECOC.mnist.models import ECOC_Hadamard_Model
from ECOC.mnist.data_handler import get_tobe_attacked_imgs
from core_attack.Attacks import JacbMDisTryAndOptimize
import os, time
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# tf.random.set_seed(19930117)
np.random.seed(19930117)
a = np.random.get_state()
np.random.set_state(a)

'''windows'''
# # load models & initialize attack
# bit_model_weights_paths = glob(
#     'D:/LableCodingNetwork/trained-models/mnist/ECOC/Hadamard16_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5')
# bit_model_weights_paths.sort(key=lambda x: int(x.split('bit_')[-1].split('\\')[0]))  # windows
# print(bit_model_weights_paths)
# HADAMARD_MATRIX = np.load(r'D:\LableCodingNetwork\ecoc_hadamard_mnist\hadamard16.npy')
# # load to-be attacked images
# index_path = r'D:\LableCodingNetwork\trained-models\mnist\ECOC\tobe_attacked_200_mnist_testset_July3_index.npy'
# error_pattern_path = r'D:\LableCodingNetwork\trained-models\mnist\ECOC\tobe_attacked_200_mnist_testset_July3_error_pattern.npy'
# tobe_attacked_clean_images, ground_labels, ground_hadamard_codewords =\
#     get_tobe_attacked_imgs(hadamard_matrix=HADAMARD_MATRIX,
#                            mnist_testset_index=np.load(index_path))
# error_patterns = np.load(error_pattern_path)
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

''' linux '''
tf.compat.v1.disable_eager_execution()
# load models & initialize attack
bit_model_weights_paths = glob('Model/mnist/model_weight/Hadamard16_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5')
bit_model_weights_paths.sort(key=lambda x: int(x.split('bit_')[-1].replace("\\", "/").split('/')[0]))
print(bit_model_weights_paths)
HADAMARD_MATRIX = np.load('ECOC/mnist/hadamard16.npy')
index_path = 'Dataset/mnist/3rd_tobe_attacked_mnist_testset_index.npy'
error_path = 'Dataset/mnist/3rd_tobe_attacked_mnist_testset_error_pattern.npy'
tobe_attacked_clean_images, ground_labels, ground_hadamard_codewords = \
    get_tobe_attacked_imgs(hadamard_matrix=HADAMARD_MATRIX,
                           mnist_testset_index=np.load(index_path))
error_patterns = np.load(error_path)


# initialize model and attack
ecoc_model = ECOC_Hadamard_Model(bit_model_weights_paths, HADAMARD_MATRIX[:10,:])

Attack = JacbMDisTryAndOptimize(ecoc_model)


# small test to see if error patterns are ok
assert np.all(np.sum(error_patterns, axis=1) == 6)

def JacbMDisTryAndOptimize_ecoc_mnist(results_save_folder, **kwargs):
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)
    record_save_file = os.path.join(results_save_folder, 'record.txt')


    # loop imgs to attack
    success = []
    distances_psnr = []
    time_costs = []
    iterations = []

    for idx, (clean_img, ground_mnist, ground_codeword, error) in enumerate(zip(tobe_attacked_clean_images, ground_labels, ground_hadamard_codewords, error_patterns)):
        if idx > 200:
            break
        clean_img = clean_img[np.newaxis]
        assert np.prod(ecoc_model.predict(clean_img, output_type = 'codeword') == ground_codeword)
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
        # save perturbetions at each iter for later checking
        # perturbetion_record_path = os.path.join(results_save_folder, 'perturbetion_records/')
        # if not os.path.exists(perturbetion_record_path):
        #    os.makedirs(perturbetion_record_path)
        # np.save(os.path.join(perturbetion_record_path, 'number{}img_cleanL2{}_perturbetion_records.npy'.format(idx, np.sqrt(np.sum(np.square(clean_img))))), adv.other_records['perturbetion_records'])

        # print('One image consumed time:', time.clock() - _start)
        flag = adv is not None
        success.append(flag)
        # print('Success = {}, psnr = {}\n other records={}\n\n'.format(flag, adv.best_distance, adv.other_records))
        with open(record_save_file, 'a') as f:
            f.write('Success = {}, psnr = {}\n num_iter = {}, cost {} seconds\n\n'.format(flag, psnr, iteration, time_cost))
        if flag:
            distances_psnr.append(psnr)
            iterations.append(iteration)
            np.save(os.path.join(results_save_folder, 'number{}img_cleanL2{}.npy'
                                 .format(idx, np.sqrt(np.sum(np.square(clean_img))))), np.squeeze(adv))

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
        'normalize_direction': True,
        'step_size': step_size,
        'max_iteration': max_iterations,
        'confidence': confidence
    }

    path = \
        os.path.join(os.getcwd(), 'ECOC/mnist/STP={}MI{}Conf{}/')\
            .format(
            round(kwargs['step_size'],2),
            kwargs['max_iteration'],
            kwargs['confidence'],
        )
    asr, ave_psnr, ave_iteration = JacbMDisTryAndOptimize_ecoc_mnist(path, **kwargs)
    print(path)
            # print('asr = ', asr, 'psnr = ', ave_psnr)