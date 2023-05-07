import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, DataLoader
import random
import numpy as np
import datetime
from torchvision.models import resnet18, vgg16

import argparse
from argparse import ArgumentParser

import pickle



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, choices = [5, 10, 15, 20])  
parser.add_argument('--total_rounds', type=int, default=30)
parser.add_argument('--Label_Initialize', type=int, default = 20, choices = [20, 30, 40])
parser.add_argument('--model', type=str, default='resnet18', choices = ['vgg16', 'resnet18'])
parser.add_argument('--dataset', type=str, default='CIFAR10', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--num_repeats', type=int, default=10)
parser.add_argument('--acquisition', type=str, default='BADGE', choices=['random', 'GLISTER', 'CoreSet', 'BADGE'])
main_args = parser.parse_args() #args=[]

DATASET_SIZES = {
    'MNIST': 28,
    'SVHN': 32,
    'CIFAR10': 32,
    'USPS': 16
}

DATASET_NCLASSES = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    'CIFAR10': 10,
    'SVHN': 10,
    'STL10': 10,
    'LSUN': 10,
    'tiny-ImageNet': 200
}




dtst = main_args.dataset

src_size = main_args.Label_Initialize

# Load MNIST/CIFAR in 3channels (needed by torchvision models)
src_dataset = main_args.dataset
src_dataset2 = 'SVHN'
target_dataset = main_args.dataset
resize = DATASET_SIZES[main_args.dataset]
num_classes = DATASET_NCLASSES[main_args.dataset]

print('Dataset: {} Acquisition: {}'.format(src_dataset, main_args.acquisition))


import otdd
import sys
# sys.path.insert(0, 'otdd/')
from otdd.otdd.pytorch.datasets import load_imagenet, load_torchvision_data
from otdd.otdd.pytorch.distance import DatasetDistance, FeatureCost
from otdd.otdd.pytorch.datasets_active_learn import load_torchvision_data_active_learn, LabeledToUnlabeledDataset

# import distil
import os
os.makedirs('models/{}'.format(main_args.dataset), exist_ok = True)
base_dir = "models/{}".format(main_args.dataset)

import sys
# sys.path.append('distil/')
import distil
from distil.active_learning_strategies import GLISTER, BADGE, FASS, EntropySampling, RandomSampling, CoreSet, BatchBALDDropout, LeastConfidenceSampling

# from distil.active_learning_strategies.random_sampling import RandomSampling   # All active learning strategies showcased in this example
from distil.utils.models.resnet import ResNet18                                                 # The model used in our image classification example
from distil.utils.train_helper import data_train      # A utility training class provided by DISTIL



load_data_dict = {
    'vgg16': vgg16(num_classes=10),  # Set `pretrained=True` to use the pre-trained weights
    'resnet18': ResNet18(num_classes=10)
}


# print("CUDA is available:", torch.cuda.is_available())

source_data = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=64, to3channels=True, Label_Initialize = src_size, dataloader_or_not = False, maxsize=5000)
Labeled = source_data['Labeled']
Unlabeled = source_data['Unlabeled']
test = source_data['test']


def train_one(acquisition_type = 'BADGE', batch_size = main_args.batch_size, Labeled = Labeled, Unlabeled = Unlabeled, test = test, model = load_data_dict[main_args.model], n_class = num_classes, n_rounds = main_args.total_rounds, repeat = False, trial_num = 0):
    
    if acquisition_type == 'random':
        strategy_args = {'batch_size' : batch_size, 'device' : 'cuda'}  #Budget per round
        strategy = RandomSampling(Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)

    elif acquisition_type == 'GLISTER':
        strategy_args = {'lr':0.05}
        strategy = GLISTER( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)
   
    elif acquisition_type == 'CoreSet':
        strategy_args = {}
        strategy = CoreSet( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)

    elif acquisition_type == 'BADGE':
        strategy_args = {}
        strategy = BADGE( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)
    
    # elif acquisition_type == 'FASS':
    #     strategy_args = {'uncertainty_measure': 'least_confidence'}
    #     strategy = FASS( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)
    
    # elif acquisition_type == 'LeastConf':
    #     strategy_args = {}
    #     alg = LeastConfidenceSampling( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)

        


    # Use the same training parameters as before
    args = {'n_epoch':100, 'lr':float(0.001), 'batch_size':20, 'max_accuracy':0.99, 'optimizer':'adam'} 
    dt = data_train(Labeled, model, args)

    # Update the model used in the AL strategy with the loaded initial model
    strategy.update_model(model)

    # Get the test accuracy of the initial model
    acc = np.zeros(n_rounds)
    acc[0] = dt.get_acc_on_set(test)
    print('Initial Testing accuracy:', round(acc[0]*100, 2), flush=True)

    # User Controlled Loop
    for rd in range(1, n_rounds):
        print('-------------------------------------------------')
        print('Round', rd) 
        print('-------------------------------------------------')

    # Use select() to obtain the indices in the unlabeled set that should be labeled
        # organ_amnist_full_train.transform = organ_amnist_test_transform       # Disable augmentation while selecting new points as to not interfere with the strategies
        idx = strategy.select(batch_size)
        # organ_amnist_full_train.transform = organ_amnist_training_transform   # Enable augmentation

    # Add the selected points to the train set. The unlabeled set shown in the next couple lines 
    # already has the associated labels, so no human labeling is needed. Again, this is because 
    # we already have the labels a priori. In real scenarios, a human oracle would need to provide 
    # then before proceeding.
        Labeled = ConcatDataset([Labeled, Subset(Unlabeled, idx)])
        Remaining_unlabeled_idx = list(set(range(len(Unlabeled))) - set(idx))
        Unlabeled = Subset(Unlabeled, Remaining_unlabeled_idx)

        print('Number of Labeled points -', len(Labeled))

        # Update the data used in the AL strategy and the training class
        strategy.update_data(Labeled, LabeledToUnlabeledDataset(Unlabeled))
        dt.update_data(Labeled)

        # Retrain the model and update the strategy with the result
        model = dt.train()
        strategy.update_model(model)

        # Get new test accuracy
        acc[rd] = dt.get_acc_on_set(test)
        print('Testing accuracy:', round(acc[rd]*100, 2), flush=True)

    print('Training Completed')

    if repeat == True:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        file_name = "Accuracy_{}_For_{}_Total_Rounds_{}.txt".format(timestamp, acquisition_type, main_args.total_rounds)
        with open(os.path.join(base_dir,file_name), 'w') as f:
            for item in acc:
                f.write("%s\n" % item)
        return acc
        
    # Lastly, we save the accuracies in case a comparison is warranted.
    with open(os.path.join(base_dir,'Acquisition Type {} for Dataset {}.txt'.format(acquisition_type, main_args.dataset)), 'w') as f:
        for item in acc:
            f.write("%s\n" % item)
    return acc


def training_loop(acquisition_type = 'CoreSet', trials = main_args.num_repeats):
    trials_acc = []
    for trial in range(trials):
        one_trial_acc = train_one(acquisition_type=acquisition_type, repeat = True, trial_num = trial)
        trials_acc.append(one_trial_acc)
        print('Trial {}: Accuracy {}'.format(trial+1, one_trial_acc))
        # # Get the current date and time
        # now = datetime.datetime.now()

        # # Format the timestamp as a string (e.g., '2023-05-05_15-30-45')
        # timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

        # # Concatenate the timestamp with the file name
        # file_name = "model_{}_For_{}_Trial_{}.pt".format(timestamp, acquisition_type, trial+1)

        # # Save the model
        # torch.save(model.state_dict(), file_name)
    with open(os.path.join(base_dir, 'Acquisition Type {} for Dataset {} Trials {}.pkl'.format(acquisition_type, main_args.dataset, trials)), 'wb') as fp:
        pickle.dump(trials_acc, fp)
    return trials_acc
        
            
#train_one(acquisition_type='BatchBALD')       
# training_loop(acquisition_type='CoreSet')



training_loop(acquisition_type = main_args.acquisition)



# # Load data
# loaders_src  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=200)[0]
# loaders_tgt  = load_torchvision_data('SVHN',  valid_size=0, resize = 28, maxsize=200)[0]

# # Instantiate distance
# dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
#                           inner_ot_method = 'exact',
#                           debiased_loss = True,
#                           p = 2, entreg = 1e-1,
#                           device='cpu')

# d = dist.distance(maxsamples = 1000)
# print(f'OTDD(MNIST,MNIST)={d:8.2f}')


# import csv

# import torch
# from torchvision.models import resnet18

# from otdd.pytorch.datasets import load_torchvision_data
# from otdd.pytorch.distance import DatasetDistance, FeatureCost



# tgt_size = 4000
# Total_rounds = args.total_rounds

# loaders_src = load_torchvision_data(src_dataset, resize=resize, batch_size = batch_size, to3channels=True, maxsize=5000)[0]
# loaders_src2 = load_torchvision_data(src_dataset2, resize=28, maxsize=src_size)[0]
# loaders_tgt = load_torchvision_data(target_dataset, resize=resize, batch_size = 100, to3channels=True, maxsize=tgt_size)[0]

# # # Embed using a pretrained (+frozen) resnet
# # embedder = resnet18(pretrained=True).eval()
# # embedder.fc = torch.nn.Identity()
# # for p in embedder.parameters():
# #     p.requires_grad = False

# # # Here we use same embedder for both datasets
# # feature_cost = FeatureCost(src_embedding = embedder,
# #                            src_dim = (3,28,28),
# #                            tgt_embedding = embedder,
# #                            tgt_dim = (3,28,28),
# #                            p = 2,
# #                            device='cuda')

# # dist = DatasetDistance(loaders_src2['train'], loaders_tgt['train'],
# #                           inner_ot_method = 'exact',
# #                           debiased_loss = True,
# #                           feature_cost = feature_cost,
# #                           sqrt_method = 'spectral',
# #                           sqrt_niters=10,
# #                           precision='single',
# #                           p = 2, entreg = 1e-1,
# #                           device='cuda')

# # d = dist.distance(maxsamples = 10000)
# # print(f'Embedding: OTDD(SVHN,MNIST)={d:8.2f}')






# def train(model, loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)

# def test(model, loader, criterion, device, test_size = tgt_size):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             total_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     return total_loss / len(loader), correct / test_size

# def training_MNIST_random(train_loader, test_loader, num_epochs = 50, Total_rounds = Total_rounds, model = model):
#     # model = resnet18(num_classes=10)  # Set the number of output classes to 10 for MNIST
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     test_accuracies = []  # Create a list to store test accuracies


#     for _ in range(Total_rounds):
#         for epoch in range(num_epochs):
#             train_loss = train(model, train_loader, criterion, optimizer, device)
#             test_loss, accuracy = test(model, test_loader, criterion, device)
#         print(f"Round {_ + 1}/{Total_rounds}\nTrain Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
#         test_accuracies.append(accuracy)
        
#     with open("test_accuracies for model {} random with {} data {} batch size.csv".format(args.model, args.dataset, args.batch_size), "w", newline="") as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(["Round", "Test Accuracy"])
#         for i, acc in enumerate(test_accuracies):
#             csv_writer.writerow([i + 1, acc])


# print('Length for Training Dataloaders {}'.format(len(loaders_src['train'])), 'Length for Testing Dataloader {}'.format(len(loaders_tgt['train'])))

# training_MNIST_random(train_loader = loaders_src['train'], test_loader = loaders_tgt['train'])




