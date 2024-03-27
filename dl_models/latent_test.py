import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import os
import json
import math
import getopt, argparse
import latent.latent_util as latent_util


sys.path.append('../')

from trainer import *
from fingerprint_dataset import FingerprintDataset
from multiple_finger_datasets import *
from embedding_models import *
from common_filepaths import *

import wandb

output_dir = '/home/albert/crystal/LatentFingerprintMatching/dl_models/latent/inspect_images/res' 

# Pre: parameters are 2 1D tensors
def euclideanDist(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(0)

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

    # One-sided Welch's t-test that diff-person pairs are more dissimilar than same-person pairs
    # welch_t, p_val = ttest_ind(_01_dist, _02_dist, equal_var=False, alternative='less')

    return acc, fpr, tpr, auc, threshold# , welch_t, p_val

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


def main(args, cuda):    
    datasets = args.datasets.split()
    print("datasets", datasets)
    possible_fgrps = args.possible_fgrps.split()
    assert set(possible_fgrps).issubset(set(ALL_FINGERS))

    the_name = '_'.join([path[:len(path) if path[-1] != '/' else -1].split('/')[-1] for path in datasets])
    print('Name of this dataset:', the_name)

    test_dir_paths = [os.path.join(x, 'test') for x in datasets]

    testing_dataset = MultipleFingerDataset(fingerprint_dataset=FingerprintDataset(test_dir_paths, train=False),\
        num_anchor_fingers=1, num_pos_fingers=1, num_neg_fingers=1,\
        SCALE_FACTOR=args.scale_factor,\
        diff_fingers_across_sets=args.diff_fingers_across_sets_train, diff_fingers_within_set=True,\
        diff_sensors_across_sets=args.diff_sensors_across_sets_train, same_sensor_within_set=True, \
        acceptable_anchor_fgrps=possible_fgrps, acceptable_pos_fgrps=possible_fgrps, acceptable_neg_fgrps=possible_fgrps)
    print(len(testing_dataset))

    test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=16)
    print(len(test_dataloader))

    weights_path = "/home/albert/crystal/LatentFingerprintMatching/latent-output/results/weights_2024-03-25_00:21:18.pth"
    latent_weights_path = "/home/albert/crystal/LatentFingerprintMatching/latent-output/results/weights_2024-03-25_00:21:18_latent.pth"

    embedder = EmbeddingNet()
    embedder.load_state_dict(torch.load(weights_path, map_location=torch.device(cuda)))

    embedder.eval()
    embedder.to(cuda)

    latent_embedder = EmbeddingNet()
    latent_embedder.load_state_dict(torch.load(latent_weights_path, map_location=torch.device(cuda)))

    latent_embedder.eval()
    latent_embedder.to(cuda)

    print("model loaded!!!!!")

    data_iter = iter(test_dataloader)

    total_d01, total_d02 = 0, 0
    d01_distances = []
    d02_distances = []

    with torch.no_grad():
        for i in tqdm(range(len(test_dataloader))):
            test_images, test_labels, test_filepaths = next(data_iter)
            anchor_image = test_images[0].to(cuda)
            pos_image = test_images[1].to(cuda)
            neg_image = test_images[2].to(cuda)

            embedder_anchor = torch.flatten(latent_embedder(anchor_image))
            embedder_pos = torch.flatten(embedder(pos_image))
            embedder_neg = torch.flatten(embedder(neg_image))
            d01 = euclideanDist(embedder_anchor, embedder_pos)
            d02 = euclideanDist(embedder_anchor, embedder_neg)

            total_d01 += d01
            total_d02 += d02

            d01_distances.append(d01)
            d02_distances.append(d02)


            # print()
            assert len(test_images) == 3
            # print("test batch...", len(test_filepaths))
            # print("test labels...", test_labels)
            # print("test filepaths...", test_filepaths)
            # embedder()
            # embedding_anchor = torch.flatten(embedder(curr_anchor))
    total_d01 /= len(test_dataloader)
    total_d02 /= len(test_dataloader)

    print("total_d01", total_d01)
    print("total_d02", total_d02)
    accs, fpr, tpr, auc, threshold = get_metrics(d01_distances, d02_distances)

    plot_roc_auc(fpr=fpr, tpr=tpr, \
        dataset_name='latent302', weights_name='w25', \
        num_anchors='1', num_pos='1', num_neg='1')


    return 





    # CLEAR CUDA MEMORY
    # https://stackoverflow.com/questions/54374935/how-to-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # LOG TRAINING DATA
    print('Training data: {}\n'.format(train_dir_paths))

    # CREATE EMBEDDER
    embedder = EmbeddingNet(pretrained=False)

    # load saved weights!
    if args.pretrained_model_path:
        print('loading pretrain state dict')
        embedder.load_state_dict(torch.load(args.pretrained_model_path))
        print('successfully loaded pretrain state dict')

    pretrained_other_msg = 'pretrained on other data: {}\n'.format(args.pretrained_model_path)
    print(pretrained_other_msg)

    # CREATE TRIPLET NET
    triplet_net = TripletNet(embedder)

    # TRAIN
    optimizer = optim.Adam(triplet_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, 
                                                     eta_min=args.lr*1e-3, last_epoch=- 1, verbose=False)

    print('learning rate = {}\ntriplet loss margin = {}\n'.format(args.lr, args.tripletLoss_margin))
    print('max epochs = {}\n'.format(args.num_epochs))

    best_val_epoch, best_val_loss = 0, 0
    all_epochs, past_train_losses, past_val_losses = [0], [0], [0]

    if args.wandb_project:
        wandb.summary['model'] = str(triplet_net)
        wandb.watch(triplet_net, log='all', log_freq=500)

    best_val_epoch, best_val_loss, all_epochs, past_train_losses, past_val_losses = fit(
        train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net,
        loss_fn=nn.TripletMarginLoss(margin=args.tripletLoss_margin), optimizer=optimizer, scheduler=scheduler,
        n_epochs=args.num_epochs, cuda=device, log_interval=args.log_interval, metrics=[], 
        start_epoch=0, early_stopping_interval=args.early_stopping_interval,
        num_accumulated_batches=args.num_accumulated_batches, 
        temp_model_path=os.path.join(args.temp_model_dir, 'temp_{}.pth'.format(the_name))
    )
    print('best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))

    # SAVE MODEL
    os.makedirs(os.path.dirname(args.posttrained_model_path), exist_ok=True)
    torch.save(embedder.state_dict(), args.posttrained_model_path)

    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs(args.results_dir, exist_ok=True)
    with open('{}/results_{}.txt'.format(args.results_dir, datetime_str), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)
        fout.write('\nbest_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))
        fout.write('\nepochs: {}\ntrain_losses: {}\nval_losses: {}\n'.format(all_epochs, past_train_losses, past_val_losses))
    torch.save(embedder.state_dict(), os.path.join(args.results_dir, 'weights_{}.pth'.format(datetime_str)))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Fingerprint Matcher')
    # training loop arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num-accumulated-batches', type=int, default=1,
                        help='number of accumulated batches before weight update (default: 1)')
    parser.add_argument('--num-epochs', type=int, default=250,
                        help='number of epochs to train (default: 250)')
    parser.add_argument('--early-stopping-interval', type=int, default=85,
                        help='how long to train model before early stopping, if no improvement')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--tripletLoss-margin', type=float, default=0.2,
                        help='Margin for triplet loss (default: 0.2)')
    # model arguments
    parser.add_argument('--pretrained-model-path', type=str, default=None,
                        help='path to pretrained model (default: None)')
    parser.add_argument('--posttrained-model-path', type=str, default='/data/therealgabeguo/fingerprint_weights/curr_model.pth',
                        help='path to save the model at')
    # saving arguments
    parser.add_argument('--temp_model_dir', type=str, default='temp_weights',
                        help='where to save the temporary model weights, as the model is training')
    parser.add_argument('--results_dir', type=str, default='/data/therealgabeguo/results',
                        help='what directory to save the results in')
    # dataset arguments
    parser.add_argument('--datasets', type=str, default='/data/therealgabeguo/fingerprint_data/sd302_split /data/albert/302_latent_data_split',
                        help='where is the data stored')
    parser.add_argument('--scale-factor', type=int, default=1,
                        help='number of times to go over the dataset to create triplets (default: 1)')
    parser.add_argument('--possible-fgrps', type=str, default='01 02 03 04 05 06 07 08 09 10',
                        help='Possible finger types to use in analysis (default: \'01 02 03 04 05 06 07 08 09 10\')')
    parser.add_argument('--diff-fingers-across-sets-train', action='store_true',
                        help='Whether to force different fingers across sets in training')
    parser.add_argument('--diff-sensors-across-sets-train', action='store_true',
                        help='Whether to force different sensors across sets in training')
    parser.add_argument('--diff-fingers-across-sets-val', action='store_true',
                        help='Whether to force different fingers across sets in validation')
    parser.add_argument('--diff-sensors-across-sets-val', action='store_true',
                        help='Whether to force different sensors across sets in validation')
    # miscellaneous arguments
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300,
                        help='How many batches to go through before logging in training')
    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='fingerprint_correlation', \
                        help='database name for wandb')
        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main(args, device)