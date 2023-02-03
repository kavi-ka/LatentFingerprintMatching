"""
-> Tests the performance of any given model on any given dataset.
-> Can customize number of fingers in each testing set (e.g., anchor set, positive set, negative set).
-> Forces different sensor and different finger.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import sys
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import ttest_ind

import json

sys.path.append('../')
sys.path.append('../directory_organization')

import matplotlib
matplotlib.use("Agg")

from trainer import *
from fingerprint_dataset import *
from multiple_finger_datasets import *
from embedding_models import *
from fileProcessingUtil import *

from common_filepaths import *

# Default Values
DEFAULT_OUTPUT_ROOT = '/data/therealgabeguo/results'
DEFAULT_CUDA = 'cuda:2'

# Batch Size constant
batch_size=1 # must be 1
assert batch_size == 1

# JSON Keys
DATASET_KEY = 'dataset'
WEIGHTS_KEY = 'weights'
CUDA_KEY = 'cuda'
OUTPUT_DIR_KEY = 'output dir'
NUM_ANCHOR_KEY = 'number anchor fingers per set'
NUM_POS_KEY = 'number positive fingers per set'
NUM_NEG_KEY = 'number negative fingers per set'
SCALE_FACTOR_KEY = 'number of times looped through dataset'
EXCLUDE_SAME_FINGER_KEY = 'exclude overlapping fingers'
NUM_SAMPLES_KEY = 'number of distinct samples in dataset'
NUM_POS_PAIRS_KEY = 'number of positive pairs'
NUM_NEG_PAIRS_KEY = 'number of negative pairs'
MEAN_POS_DIST_KEY = 'average squared L2 distance between positive pairs'
STD_POS_DIST_KEY = 'std of  squared L2 distance between positive pairs'
MEAN_NEG_DIST_KEY = 'average squared L2 distance between negative pairs'
STD_NEG_DIST_KEY = 'std of  squared L2 distance between negative pairs'
ACC_KEY = 'best accuracy'
ROC_AUC_KEY = 'ROC AUC'
T_VAL_KEY = 'Welch\'s t'
P_VAL_KEY = 'p-value'

# More constants
SAME_PERSON = 0
DIFF_PERSON = 1

# Pre: parameters are 2 1D tensors
def euclideanDist(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(0)

# Inputs: (_01_dist, _02_dist) are distance between anchor and (positive, negative), repsectively
# Returns: (acccuracies, fpr, tpr, ROC AUC, threshold, welch_t, p_val)
# - accuracies are at every possible threshold
# - fpr is false positive rate at every possible threshold (padded with 0 and 1 at end)
# - tpr is true positive rate at every possible threshold (padded with 0 and 1 at end)
# - roc_auc is scalar: area under fpr (x-axis) vs tpr (y-axis) curve
# - threshold is scalar: below this distance, fingerpritnts match; above, they don't match
# - welch_t is value of Welch's two-sample t-test between same-person and diff-person pairs
# - p-val is statistical significance
def get_metrics(_01_dist, _02_dist):
    all_distances = _01_dist +_02_dist
    all_distances.sort()

    tp, fp, tn, fn = list(), list(), list(), list()
    acc = list()

    # try different thresholds
    for dist in all_distances:
        tp.append(len([x for x in _01_dist if x < dist]))
        tn.append(len([x for x in _02_dist if x >= dist]))
        fn.append(len(_01_dist) - tp[-1])
        fp.append(len(_02_dist) - tn[-1])

        acc.append((tp[-1] + tn[-1]) / len(all_distances))
    threshold = all_distances[max(range(len(acc)), key=acc.__getitem__)]

    # ROC AUC is FPR = FP / (FP + TN) (x-axis) vs TPR = TP / (TP + FN) (y-axis)
    fpr = [0] + [fp[i] / (fp[i] + tn[i]) for i in range(len(fp))] + [1]
    tpr = [0] + [tp[i] / (tp[i] + fn[i]) for i in range(len(tp))] + [1]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])

    assert auc >= 0 and auc <= 1

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]

    # Welch's t-test
    welch_t, p_val = ttest_ind(_01_dist, _02_dist, equal_var=False)

    return acc, fpr, tpr, auc, threshold, welch_t, p_val

"""
Input: 
-> matrix f2f_dist of finger by finger distances where f2f_dist[i][j][k] = 
    list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
Returns:
-> Matrix of ROC AUC per finger pair
-> Matrix of p-value per finger pair
-> Matrix of number of samples per finger pair
-> Matrix of percent of samples that were same-person per finger pair
"""
def get_finger_by_finger_metrics(finger_to_finger_dist):
    finger_to_finger_num_samples = np.zeros((10, 10))
    finger_to_finger_percent_samePerson = np.zeros((10, 10))
    finger_to_finger_roc = np.zeros((10, 10))
    finger_to_finger_p_val = np.zeros((10, 10))
    for i in range(1, 10+1):
        for j in range(1, 10+1):
            same_person_dists = finger_to_finger_dist[i][j][SAME_PERSON]
            diff_person_dists = finger_to_finger_dist[i][j][DIFF_PERSON]

            _i, _j = i - 1, j - 1

            if len(same_person_dists) == 0 or len(diff_person_dists) == 0:
                finger_to_finger_roc[_i][_j] = np.NaN
                finger_to_finger_p_val[_i][_j] = np.NaN
                finger_to_finger_num_samples[_i][_j] = np.NaN
                finger_to_finger_percent_samePerson[_i][_j] = np.NaN
                continue

            accuracies, fpr, tpr, roc_auc, threshold, welch_t, p_val = get_metrics(same_person_dists, diff_person_dists)
            
            finger_to_finger_roc[_i][_j] = roc_auc
            finger_to_finger_p_val[_i][_j] = p_val

            finger_to_finger_num_samples[_i][_j] = len(same_person_dists) + len(diff_person_dists)
            finger_to_finger_percent_samePerson[_i][_j] = len(same_person_dists) / finger_to_finger_num_samples[_i][_j]

    return finger_to_finger_roc, finger_to_finger_p_val, finger_to_finger_num_samples, finger_to_finger_percent_samePerson

def create_output_dir(output_root):
    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(output_root, datetime_str)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_filename(img_filepath):
    if type(img_filepath) is tuple or type(img_filepath) is list:
        assert len(img_filepath) == 1
        img_filepath = img_filepath[0]
    return img_filepath.split('/')[-1]

def get_int_fgrp_from_filepath(img_filepath):
    return int(get_fgrp(get_filename(img_filepath)))

# Returns whether we are doing 1-1 matching
def is_one_to_one(num_anchors, num_pos, num_neg):
    return num_anchors == 1 and num_pos == 1 and num_neg == 1

"""
Returns: (_01_dist, _02_dist, f2f_dist)
-> _01_dist containing distances between anchor and positive examples (from test_dataloader, using embedder)
-> _02_dist containing distances between anchor and negative examples
-> matrix f2f_dist of finger by finger distances where f2f_dist[i][j][k] = 
    list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
"""
def run_test_loop(test_dataloader, embedder, cuda, num_anchors, num_pos, num_neg):
    # distances between embedding of positive and negative pair
    _01_dist = []
    _02_dist = []
    # finger_to_finger_dist[i][j][k] = list of distances between FGRP (i, j) for matching status k (SAME_PERSON or DIFF_PERSON)
    finger_to_finger_dist = [[[[] for k in (SAME_PERSON, DIFF_PERSON)] for j in range(0, 10+1)] for i in range(0, 10+1)]

    data_iter = iter(test_dataloader)
    assert batch_size == 1
    for i in range(len(test_dataloader)):
        test_images, test_labels, test_filepaths = next(data_iter)
        assert len(test_images) == 3
        assert len(test_images[0]) == num_anchors and len(test_images[1]) == num_pos and len(test_images[2]) == num_neg

        # test_images is 3 (anchor, pos, neg) * N (number of sample images) * image_size (1*3*224*224)

        curr_anchor_pos_dists = []
        curr_anchor_neg_dists = []

        for i_a in range(num_anchors):
            curr_anchor = test_images[0][i_a].to(cuda)
            embedding_anchor = torch.flatten(embedder(curr_anchor))
            assert len(embedding_anchor.size()) == 1 and embedding_anchor.size(dim=0) == 512      
            anchor_filepath = test_filepaths[0][i_a]
            anchor_finger = get_int_fgrp_from_filepath(anchor_filepath)

            for i_p in range(num_pos):
                curr_pos = test_images[1][i_p].to(cuda)
                embedding_pos = torch.flatten(embedder(curr_pos))
                assert len(embedding_pos.size()) == 1 and embedding_pos.size(dim=0) == 512
                curr_anchor_pos_dists.append(euclideanDist(embedding_anchor, embedding_pos).item())
                # finger-by-finger
                pos_filepath = test_filepaths[1][i_p]
                pos_finger = get_int_fgrp_from_filepath(pos_filepath)
                finger_to_finger_dist[anchor_finger][pos_finger][SAME_PERSON].append(curr_anchor_pos_dists[-1])
                if anchor_finger != pos_finger:
                    finger_to_finger_dist[pos_finger][anchor_finger][SAME_PERSON].append(curr_anchor_pos_dists[-1])
            for i_n in range(num_neg):
                curr_neg = test_images[2][i_n].to(cuda)
                embedding_neg = torch.flatten(embedder(curr_neg))
                assert len(embedding_neg.size()) == 1 and embedding_neg.size(dim=0) == 512
                curr_anchor_neg_dists.append(euclideanDist(embedding_anchor, embedding_neg).item())
                # finger-by-finger
                neg_filepath = test_filepaths[2][i_n]
                neg_finger = get_int_fgrp_from_filepath(neg_filepath)
                finger_to_finger_dist[anchor_finger][neg_finger][DIFF_PERSON].append(curr_anchor_neg_dists[-1])
                if anchor_finger != neg_finger:
                    finger_to_finger_dist[neg_finger][anchor_finger][DIFF_PERSON].append(curr_anchor_neg_dists[-1])
        
        _01_dist.append(np.mean(curr_anchor_pos_dists))
        _02_dist.append(np.mean(curr_anchor_neg_dists))

        if i % 100 == 0:
            print('Batch (item) {} out of {}'.format(i, len(test_dataloader)))
            print('\taverage, std squared L2 distance between positive pairs {:.3f}, {:.3f}'.format(np.mean(_01_dist), np.std(_01_dist)))
            print('\taverage, std squared L2 distance between negative pairs {:.3f}, {:.3f}'.format(np.mean(_02_dist), np.std(_02_dist)))
    
    return _01_dist, _02_dist, finger_to_finger_dist

"""
Saves a ROC curve
"""
def plot_roc_auc(fpr, tpr, dataset_name, weights_name, num_anchors, num_pos, num_neg):
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}_{}_{}_{}.pdf'.format(\
        dataset_name, weights_name, num_anchors, num_pos, num_neg)))
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}_{}_{}_{}.png'.format(\
        dataset_name, weights_name, num_anchors, num_pos, num_neg)))
    plt.clf(); plt.close()

    return

"""
Returns:
-> FingerprintDataset from the_data_folder, guaranteed to be testing
-> MultipleFingerDataset from FingerprintDataset, which has triplets of anchor, pos, neg fingerprints;
    number according to scale_factor, size fo triplets according to (num_anchors, num_pos, num_neg)
-> DataLoader with batch size 1 for MultipleFingerDataset 
"""
def load_data(the_data_folder, num_anchors, num_pos, num_neg, scale_factor, exclude_same_finger):
    fingerprint_dataset = FingerprintDataset(os.path.join(the_data_folder, 'test'), train=False)
    test_dataset = MultipleFingerDataset(fingerprint_dataset, num_anchors, num_pos, num_neg, \
        SCALE_FACTOR=scale_factor, exclude_same_finger=exclude_same_finger)
    print('loaded test dataset: {}'.format(the_data_folder))
    assert batch_size == 1
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return fingerprint_dataset, test_dataset, test_dataloader

def create_finger_by_finger_plot(f2f_data, the_title, the_cmap, the_fontsize, the_fmt):
    fgrp_names = ['Right Thumb', 'Right Index', 'Right Middle', 'Right Ring', 'Right Pinky', \
        'Left Thumb', 'Left Index', 'Left Middle', 'Left Ring', 'Left Pinky']
    plt.subplots_adjust(bottom=0.22, left=0.22)
    plt.title(the_title)
    sns.heatmap(f2f_data, annot=True, xticklabels=fgrp_names, yticklabels=fgrp_names, \
        cmap=the_cmap, fmt=the_fmt, annot_kws={"fontsize":the_fontsize})
    plt.savefig(os.path.join(output_dir, '{}.pdf'.format(the_title)))
    plt.savefig(os.path.join(output_dir, '{}.png'.format(the_title)))
    plt.clf(); plt.close()
    return

def main(the_data_folder, weights_path, cuda, output_dir, num_anchors, num_pos, num_neg, \
        scale_factor=1, exclude_same_finger=True):
    print('Number anchor, pos, neg fingers: {}, {}, {}'.format(num_anchors, num_pos, num_neg))

    fingerprint_dataset, test_dataset, test_dataloader = \
        load_data(the_data_folder=the_data_folder, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg, \
        scale_factor=scale_factor, exclude_same_finger=exclude_same_finger)

    dataset_name = the_data_folder[:-1 if the_data_folder[-1] == '/' else len(the_data_folder)].split('/')[-1]
    print('dataset name:', dataset_name)
    weights_name = (weights_path.split('/')[-1])[:-4]
    print('weights name:', weights_name)

    # CREATE EMBEDDER
    embedder = EmbeddingNet()

    # LOAD MODEL

    embedder.load_state_dict(torch.load(weights_path))
    embedder.eval()
    embedder.to(cuda)

    # TEST
    assert batch_size == 1

    _01_dist, _02_dist, finger_to_finger_dist = run_test_loop(test_dataloader=test_dataloader, embedder=embedder, cuda=cuda, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg)

    # TODO: update finger-by-finger
    f2f_roc, f2f_p_val, f2f_num_samples, f2f_percent_samePerson = get_finger_by_finger_metrics(finger_to_finger_dist)
    create_finger_by_finger_plot(f2f_roc, 'Finger-to-Finger ROC AUC', the_cmap='Reds', the_fontsize=9, the_fmt='.2f')
    create_finger_by_finger_plot(f2f_p_val, 'Finger-to-Finger P-Value (Welch\'s T-Test', the_cmap='Purples', the_fontsize=6, the_fmt='.2g')
    create_finger_by_finger_plot(f2f_num_samples, 'Number of Finger-to-Finger Pairs', the_cmap='Blues', the_fontsize=9, the_fmt='g')
    create_finger_by_finger_plot(f2f_percent_samePerson, 'Proportion of Same-Person Samples', the_cmap='Greens', the_fontsize=9, the_fmt='.2f')

    # CALCULATE ACCURACY AND ROC AUC
    accs, fpr, tpr, auc, threshold, welch_t, p_val = get_metrics(_01_dist, _02_dist)

    plot_roc_auc(fpr=fpr, tpr=tpr, \
        dataset_name=dataset_name, weights_name=weights_name, \
        num_anchors=num_anchors, num_pos=num_pos, num_neg=num_neg)

    # do the output
    final_results = {
        DATASET_KEY: the_data_folder, WEIGHTS_KEY: weights_path, CUDA_KEY: cuda,
        OUTPUT_DIR_KEY: output_dir,
        NUM_ANCHOR_KEY: num_anchors, NUM_POS_KEY: num_pos, NUM_NEG_KEY: num_neg,
        SCALE_FACTOR_KEY: scale_factor,
        EXCLUDE_SAME_FINGER_KEY: exclude_same_finger,
        NUM_SAMPLES_KEY: len(fingerprint_dataset),
        NUM_POS_PAIRS_KEY: len(_01_dist), NUM_NEG_PAIRS_KEY: len(_02_dist), 
        MEAN_POS_DIST_KEY: np.mean(_01_dist), STD_POS_DIST_KEY: np.std(_01_dist),
        MEAN_NEG_DIST_KEY: np.mean(_02_dist), STD_NEG_DIST_KEY: np.std(_02_dist),
        ACC_KEY: max(accs), ROC_AUC_KEY: auc,
        T_VAL_KEY: welch_t, P_VAL_KEY: p_val
    }

    results_fname = os.path.join(output_dir, \
        'test_results_{}_{}_{}_{}_{}.json'.format(dataset_name, weights_name, num_anchors, num_pos, num_neg))
    with open(results_fname, 'w') as fout:
        fout.write(json.dumps(final_results, indent=4))

    # print output
    with open(results_fname, 'r') as fin:
        for line in fin:
            print(line.rstrip())
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parameterized_multiple_finger_tester.py')
    parser.add_argument('--dataset', '-d', help='Path to folders containing images', type=str)
    parser.add_argument('--weights', '-w', help='Path to model weights', type=str)
    parser.add_argument('--cuda', '-c', nargs='?', \
        const=DEFAULT_CUDA, default=DEFAULT_CUDA, help='Name of GPU we want to use', type=str)
    parser.add_argument('--num_fingers', '-n', nargs='?', help='Number of fingers to test', \
        const=1, default=1, type=int)
    parser.add_argument('--output_root', '-o', nargs='?', help='Root directory for output', \
        const=DEFAULT_OUTPUT_ROOT, default=DEFAULT_OUTPUT_ROOT, type=str)
    parser.add_argument('--scale_factor', '-s', nargs='?', help='Number of times to loop through the dataset to create triplets', \
        const=1, default=1, type=int)
    parser.add_argument('--exclude_same_finger', '-e', nargs='?', \
        help='True (by default) if we ban pairs with overlapping fingers, can be False iff we do 1-1 correlation experiment',
        const=True, default=True, type=bool)

    args = parser.parse_args()

    dataset = args.dataset
    weights = args.weights
    cuda = args.cuda
    num_fingers = args.num_fingers
    output_dir = create_output_dir(args.output_root)
    scale_factor = args.scale_factor
    exclude_same_finger = args.exclude_same_finger

    assert num_fingers > 0
    assert scale_factor >= 1

    main(the_data_folder=dataset, weights_path=weights, cuda=cuda, output_dir=output_dir, \
        num_anchors=num_fingers, num_pos=num_fingers, num_neg=num_fingers, \
        scale_factor=scale_factor, exclude_same_finger=exclude_same_finger)

    # TODO: test this, make sure it's doing what I want
    # TODO: have option to have same fingers in the set in multiple_finger_datasets.py