import argparse
import sys
from glob import glob
import numpy as np
from OneHot.models import get_32class_surrogate_model
from utils.readTrafficSigns import Hadamard_labels
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from core_attack.Attacks import JacobMDisTryAndOptimizeOnehot
import os, time
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow.keras

np.random.seed(19930117)
a = np.random.get_state()
np.random.set_state(a)

# TRAINED_MODEL_PATH = os.path.join(MODEL_ROOT, 'gtsrb/ECOC/Hadamard32_surrogate_weights_freeze6_bit_*/final_trained_weights.hdf5')
TEST_IMG_FOLDER = 'Dataset/gtsrb/Final_Test/Images'
TOP32GTSRB_CATEGORIES = np.load('Dataset/gtsrb/gtsrb_top32category_label.npy')
HADAMARD_MATRIX = np.load('OneHot/hadamard32.npy')
index_path = 'Dataset/gtsrb/combined_200_ACCESS_sub_GTSRB_index.npy'
error_path = 'Dataset/gtsrb/combined_200_ACCESS_sub_GTSRB_error_pattern.npy'


model_weight_path = 'Model/gtsrb/final_trained_weights.hdf5'

to_be_attack_imgs_index = np.load(index_path)
error_patterns = np.load(error_path)

# get (num_example, 32) bit-wise ground truth label/codeword
bitwise_ground_labels_list = []
for i in range(32):
    BIT_MODEL_INDEX = i + 1
    test_img_paths, test_bit_model_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES, HADAMARD_MATRIX,
                                                            BIT_MODEL_INDEX)
    bitwise_ground_labels_list.append(test_bit_model_labels)

bitwise_ground_label_matrix = np.array(bitwise_ground_labels_list)
test_img_paths, ground_index_label_accodring_gtsrb_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES,
                                                                            HADAMARD_MATRIX, 0)

to_be_attack_imgs_bit_label_matrix = bitwise_ground_label_matrix.T[to_be_attack_imgs_index]  # shape = (100,32)
to_be_attack_imgs_paths = test_img_paths[to_be_attack_imgs_index]

# initialize model and attack
model = get_32class_surrogate_model()
model.load_weights(model_weight_path)

opt = tensorflow.keras.optimizers.RMSprop(
    # learning_rate=LR,
    decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


Attack = JacobMDisTryAndOptimizeOnehot(model)

def find_imgs_according_to_targets(image_path_pool, ground_codeword_pool, targeting_codeword, num = 50):

    indexs = np.where(ground_codeword_pool == targeting_codeword)[0]
    if len(indexs) < 50:
        num = len(indexs)
    selected_indexs = np.random.choice(indexs, num, replace=False)
    selected_paths = image_path_pool[selected_indexs]
    images = np.array([img_to_array(load_img(path, target_size=(64, 64))) / 255  for path in selected_paths])
    assert images.shape == (num, 64, 64, 3)

    return images

def gtsrb_JacobMDisTryAndOptimize_onehot(record_save_file, **kwargs):
    save_folder = os.path.dirname(record_save_file)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # loop imgs to attack
    success = []
    distances_psnr = []
    max_iter_psnrs = []
    num_iters = []
    time_costs = []

    for idx, (img_path, ground_codeword, error) in enumerate(zip(to_be_attack_imgs_paths, to_be_attack_imgs_bit_label_matrix, error_patterns)):
        if idx < 0:
            continue
        # if idx > 25:
        #     break
        clean_img = img_to_array(load_img(img_path, target_size=(64, 64)))[np.newaxis] / 255

        # error pattern indicates as -1(error) and +1(clear)
        target_undecoded = error * ground_codeword
        clean_class = np.argmax(np.matmul(ground_codeword, HADAMARD_MATRIX.T), axis=-1)
        target_class = np.argmax(np.matmul(target_undecoded, HADAMARD_MATRIX.T), axis=-1)
        if np.equal(clean_class, target_class):
            target_class = np.mod(clean_class+1, len(ground_codeword))

        # find_imgs_according_to_targets
        target_images = find_imgs_according_to_targets(test_img_paths, ground_index_label_accodring_gtsrb_labels, target_class)
        target_logits = np.mean(np.concatenate(
            [Attack._logits_func([one_target_imgs[np.newaxis]])[0] for one_target_imgs in target_images], axis=0),axis=0)

        # print('Attacking number {} img {}\n'
        #       'clean_class is {}\n'
        #       'targeting class is {}\n'
        #       .format(idx, img_path,
        #               clean_class,
        #               target_class))
        with open(record_save_file, 'a') as f:
            f.write('Attacking number {} img {}\n'
              'clean_class is {}\n'
              'targeting class is {}\n'
              .format(idx, img_path,
                      clean_class,
                      target_class))

        # print(f"Starting attack on image #{idx} - {os.path.basename(img_path)}")
        _start = time.perf_counter()
        adv_img, psnr, iteration, _ = Attack.run_attack(clean_img, targeting_output=target_logits, **kwargs)
        cost_time = time.perf_counter() - _start
        time_costs.append(cost_time)
        flag = adv_img is not None
        success.append(flag)
        # print(f"Attack ended for image #{idx} after {cost_time} seconds (success={flag})")

        # print('Success = {}, psnr = {}\n num iter={}, cost {} seconds\n\n'.format(flag, psnr, iteration, cost_time))
        with open(record_save_file, 'a') as f:
            f.write('Success = {}, psnr = {}\n num iter={}, cost {} seconds\n\n'.format(flag, psnr, iteration, cost_time))
        if flag:
            distances_psnr.append(psnr)
            np.save(os.path.join(save_folder, os.path.basename(img_path).replace('.ppm', '.npy')),
                                  np.squeeze(adv_img))
            num_iters.append(iteration)

    asr = success.count(True) / len(success)
    ave_psnr = np.mean(distances_psnr)
    ave_iteration = np.mean(num_iters)
    ave_time_cost = np.mean(time_costs)

    print('ASR = {}, ave PSNR = {}, ave num_iter = {}, ave time consumption = {}'.format(asr, ave_psnr, ave_iteration, ave_time_cost))
    with open(record_save_file, 'a') as f:
        f.write('\n\nASR = {}, ave PSNR = {}, ave num_iter = {}, ave time consumption = {}'.format(asr, ave_psnr, ave_iteration, ave_time_cost))
        f.write('index_path = {}\nerror_path = {}\nkwargs = {}\n'.format(index_path, error_path,kwargs))
    print(kwargs)
    return asr, ave_psnr, ave_iteration


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
        os.path.join(os.getcwd(), 'OneHot/STP={}MI{}Conf{}/record.txt')\
            .format(

            kwargs['step_size'],
            kwargs['max_iteration'],
            kwargs['confidence'],
        )
    asr, ave_psnr, ave_iteration = gtsrb_JacobMDisTryAndOptimize_onehot(path, **kwargs)
    print(path)
