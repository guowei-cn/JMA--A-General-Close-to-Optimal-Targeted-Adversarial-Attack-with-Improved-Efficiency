import argparse
import numpy as np
from keras.models import load_model
import os, time
# import sys
# print(sys.path)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from core_attack.Attacks import JacobMDisTryAndOptimizeMultiLabel
os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''one set attack'''

import matplotlib.pyplot as plt
def draw_AE(AE, clean_img, ground_label, target_label, psnr, save_name):
    perturb = 10 * (AE - clean_img) # todo check the range of image
    # Show subplots | shape: (1,3)
    image_l = [clean_img, perturb, AE]
    title_l = [str(ground_label).replace(" ", ""), 'noise (10*) with {:.3f}'.format(psnr), str(target_label).replace(" ", "")]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(image_l[i])
        # plt.colorbar()
        plt.title('{}'.format(title_l[i]))

    # plt.tight_layout()
    # plt.show()
    plt.savefig(save_name)

def run_one_set_attack(record_save_path, **kwargs):
    save_folder = record_save_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    record_save_file = os.path.join(save_folder, 'record.txt')
    # loop imgs to attack
    success = []
    hamming_losses = []
    distances_psnr = []
    iterations = []
    time_costs = []
    J_ranks_statics = np.empty((1,4))  #
    Jacob_0 = 0
    for idx, (clean_img, ground_label, target_label, imgID) in enumerate(zip(x_cleans, y_trues, y_tars, x_idfs)):
        if idx < 0:
            continue
        if idx > 200:
            break
        # use {-1,1} instead of {0,1} to indicate label
        if 0 in target_label:
            target_label = 2 * target_label - 1
            ground_label = 2 * ground_label - 1
        # print('Attacking number {} img ID-{}\n'
        #                     'ground label is {}\n'
        #                     'targeting label is {}\n'
        #                     .format(idx, imgID,
        #                             ground_label,
        #                             target_label))
        with open(record_save_file, 'a') as f:
            f.write('Attacking number {} img ID-{}\n'
                    'ground label is {}\n'
                    'targeting label is {}\n'
                    .format(idx, imgID,
                            ground_label,
                            target_label))

        kwargs['record_file'] = os.path.join(save_folder, imgID + '_debugInfo.txt')
        kwargs['verbos'] = False
        _start = time.time()
        attack_results = Attack.run_attack(clean_img[np.newaxis], targeting_output=target_label, **kwargs)
        time_cost = time.time() - _start
        time_costs.append(time_cost)
        # print('One image consumed time:', time.clock() - _start)

        # read attack_results
        adv = attack_results['adv']
        flag = attack_results['is_full_success']

        hamming_loss = attack_results['hamming_loss']
        J_rank_obs = attack_results['JacRank']
        psnr = attack_results['psnr']
        iteration = attack_results['iteration']
        # AE = attack_results['AE']

        J_ranks_statics = np.vstack((J_ranks_statics,
                                     J_rank_obs))  # J_rank_obs is of shape real_iters * 4, and statics if of (num_img * each_real_iter) * 4

        success.append(flag)

        # print('Success = {}, psnr = {}\n num_iter = {}, cost {} seconds\n\n'.format(flag, psnr, iteration, time_cost))
        with open(record_save_file, 'a') as f:
            f.write(
                'Full Success = {}, hamming loss = {}, psnr = {}\n num_iter = {}, cost {} seconds\n'
                'Jrank at first 5 iterations = {}\n\n'.format(flag, hamming_loss, psnr, iteration, time_cost, J_rank_obs[:5]))
        if flag:
            iterations.append(iteration)
            distances_psnr.append(psnr)
            hamming_losses.append(hamming_loss)
            np.save(os.path.join(save_folder, imgID + '_AdvOpt.npy'), np.squeeze(adv))
        else:
            np.save(os.path.join(save_folder, imgID + '_AdvSubOpt.npy'), np.squeeze(adv))

    # initial statis analysis of J_ransss

    asr = success.count(True) / len(success) if success else 0
    ave_hamming_loss = np.mean(hamming_losses)
    ave_psnr = np.mean(distances_psnr)
    ave_iteration = np.mean(iterations)
    ave_time_consumption = np.mean(time_costs)


    J_ranks_statics = np.delete(J_ranks_statics, 0, 0)
    try: ave_Jrank_each_itera = np.mean(np.array([J_ranks_statics[x:x+5] for x in range(0, J_ranks_statics.shape[0], 200)]), 0)[:,0]
    except: ave_Jrank_each_itera = np.mean(
        np.array([J_ranks_statics[x:x + 5] for x in range(0, J_ranks_statics.shape[0], 200)]), 0)[:, 0]
    # ave_Jrank_each_itera = np.mean(J_ranks_statics, axis=0)
    np.save(os.path.join(save_folder, 'allImageJrankDistribution.npy'), np.array(J_ranks_statics))

    print('ASR = {}, ave_hamming_loss = {}, ave PSNR = {}, ave num_iter = {}, ave time consumption = {}\n'
          # 'ave Jrank for fisrt 5 iteration = {}'
          .format(asr, ave_hamming_loss, ave_psnr, ave_iteration, ave_time_consumption,
                                                        # ave_Jrank_each_itera[:5]
                                                        ))

    with open(record_save_file, 'a') as f:
        f.write('\n\nASR = {}, ave_hamming_loss = {}, ave PSNR = {}, ave num_iter = {}, ave time consumption = {}\n'
          'ave Jrank for fisrt 5 iteration = {}'
          .format(asr, ave_hamming_loss, ave_psnr, ave_iteration, ave_time_consumption,
                                                        ave_Jrank_each_itera[:5]
                                                        ))
    print(kwargs)
    return asr, ave_hamming_loss, ave_psnr, ave_iteration


def multi_label_acc_metric(y_true, y_pred):
    return 1


def mAP_metric(y_true, y_pred):
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', help='model path', default='..\MultiLabel\model4voc2012.h5')
    parser.add_argument('-s', help='step size', type=float, default=0.5)
    parser.add_argument("-i", help="max iteration", default=200,
                        type=int)
    parser.add_argument('-l', default='Real',
                        help='target label type: Real, 5, 10, 15, 20')
    parser.add_argument('-nbs', help='number of binary search', type=int, default=6)
    parser.add_argument('-c', help='confidence', type=int, default= 0)
    args = parser.parse_args()
    ''' ini attack'''
    # folder and path
    # load model
    model = load_model('Model/voc2012/model4voc2012.h5',
                       custom_objects={"multi_label_acc_metric": multi_label_acc_metric, "mAP_metric": mAP_metric})
    Attack = JacobMDisTryAndOptimizeMultiLabel(model)

    ''' get to be attacked images'''
    x_ids = np.load('Dataset/voc2012/correct_predict_index.npy')
    y_trues = np.load('Dataset/voc2012/testset_ytrue.npy')[x_ids]
    y_tars = np.load('Dataset/voc2012/y_tars_{}error.npy'.format(args.l))

    x_cleans = np.load('Dataset/voc2012/x_cleans.npy')
    x_idfs = np.load('Dataset/voc2012/x_idfs.npy')


    max_iterations = args.i
    binary_search_num = args.nbs
    confidence = args.c
    kwargs = {
        'step_size': args.s,
        'max_iteration': int(max_iterations),
        'confidence': confidence,
        'verbose': False,
        'binary_search_num': binary_search_num
    }
    path = os.path.join(os.getcwd(), 'MultiLabel/targetLabel_{}_stepSize_{}_maxIter_{}_Conf_{}_bs_num_{}'.format(
        args.l,
        kwargs['step_size'],
        kwargs['max_iteration'],
        kwargs['confidence'],
        kwargs['binary_search_num']
    ))
    # if os.path.exists(path):
    #     continue
    print('attacking setting:{}'.format(kwargs))
    asr, ave_hamming_loss, ave_psnr, ave_iteration = run_one_set_attack(path, **kwargs)
    print(path)

