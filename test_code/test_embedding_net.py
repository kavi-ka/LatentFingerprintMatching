import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import os

sys.path.append('../dl_models')
#sys.path.append('../siamese-triplet')
sys.path.append('../')

from fingerprint_dataset import FingerprintDataset
from siamese_datasets import SiameseDataset, TripletDataset
from embedding_models import EmbeddingNet, SiameseNet, TripletNet
from common_filepaths import DATA_FOLDER

print('pair loading test\n')

training_dataset = SiameseDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
train_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True)

num_positive_examples = 0
for i in range(30):#range(len(training_data)):
    train_images, train_label, train_filepaths = next(iter(train_dataloader))
    num_positive_examples += train_label
    print(train_filepaths, train_label)

print(num_positive_examples)

print('\ntriplet loading test\n')

batch_size=4

triplet_dataset = TripletDataset(FingerprintDataset(os.path.join(DATA_FOLDER, 'train'), train=True))
triplet_dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

num_positive_examples = 0

triplet_net = TripletNet(EmbeddingNet())

# distances between embedding of positive and negative pair
_01_dist = []
_02_dist = []

dist = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

for i in range(10):#range(len(training_data)):
    test_images, test_labels, test_filepaths = next(iter(triplet_dataloader))
    
    embeddings = [torch.reshape(e, (batch_size, e.size()[1])) for e in triplet_net(*test_images)]
    # embeddings.shape[0] is (anchor, pos, neg); embeddings.shape[1] is batch size; embeddings.shape[2] is embedding length
    print([embedding.size() for embedding in embeddings])
    
    for batch_index in range(batch_size):
        _01_dist.append(dist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
        _02_dist.append(dist(embeddings[0][batch_index], embeddings[2][batch_index]).item())

    print(test_filepaths, test_labels)

#print(_01_dist[0].size())
#print(_02_dist[0].size())

print(len(_01_dist))
print(len(_02_dist))

print('average cosine sim between matching pairs:', sum(_01_dist) / len(_01_dist))
print('average cosine sim between non-matching pairs:', sum(_02_dist) / len(_02_dist))
