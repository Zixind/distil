import torch
import logging
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
from torchvision.models import resnet18, vgg16

import argparse
from argparse import ArgumentParser

import pickle
import os
import datetime

from NN_classification import SetTransformer_OT, DeepSet_OT, DeepSet
from torch.utils.tensorboard import SummaryWriter

import sys
# sys.path.append('distil/')
import distil
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, CoreSet, BatchBALDDropout, LeastConfidenceSampling

# from distil.active_learning_strategies.random_sampling import RandomSampling   # All active learning strategies showcased in this example
from distil.utils.models.resnet import ResNet18                                                 # The model used in our image classification example
from distil.utils.train_helper import data_train      # A utility training class provided by DISTIL



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20, choices = [5, 10, 15, 20])  
parser.add_argument('--total_rounds', type=int, default=30)
parser.add_argument('--Label_Initialize', type=int, default = 20, choices = [20, 30, 40])
parser.add_argument('--model', type=str, default='resnet18', choices = ['vgg16', 'resnet18'])
parser.add_argument('--dataset', type=str, default='MNIST', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--num_repeats', type=int, default=10)
parser.add_argument('--acquisition', type=str, default='BADGE', choices=['random', 'GLISTER', 'CoreSet', 'BADGE'])
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
parser.add_argument('--OT_distance', type=int, default=1, choices=[1, 0])
parser.add_argument('--Net_trained', type=int, default=80, choices=[20,50,80,100])
parser.add_argument('--sample_size', type=int, default=5, choices=[5, 10, 20, 30, 100, 50, 80, 70, 120])
main_args = parser.parse_args()

DATASET_SIZES = {
    'MNIST': 28,
    'SVHN': 32,
    'CIFAR10': 28,
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

Feature_Cost_dim = {
    'MNIST': (3, 28, 28),
    'CIFAR10': (3, 28, 28),
    'SVHN': (3, 32, 32)       
}

in_dims = {
    'MNIST': int(28*28),
    'CIFAR10': int(28*28),
    'SVHN': int(32*32)
}


dtst = main_args.dataset

src_size = main_args.Label_Initialize

# Load MNIST/CIFAR in 3channels (needed by torchvision models)
src_dataset = main_args.dataset
target_dataset = main_args.dataset
resize = DATASET_SIZES[main_args.dataset]
num_classes = DATASET_NCLASSES[main_args.dataset]


import otdd
import sys
# sys.path.insert(0, 'otdd/')
from otdd.otdd.pytorch.datasets import load_imagenet, load_torchvision_data
from otdd.otdd.pytorch.distance import DatasetDistance, FeatureCost
from otdd.otdd.pytorch.datasets_active_learn import load_torchvision_data_active_learn, LabeledToUnlabeledDataset

# import distil
import os
os.makedirs('models', exist_ok = True)
base_dir = "models"

import sys
# sys.path.append('distil/')
import distil
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, CoreSet, BatchBALDDropout, LeastConfidenceSampling

# from distil.active_learning_strategies.random_sampling import RandomSampling   # All active learning strategies showcased in this example
from distil.utils.models.resnet import ResNet18                                                 # The model used in our image classification example
from distil.utils.train_helper import data_train      # A utility training class provided by DISTIL



load_data_dict = {
    'vgg16': vgg16(num_classes=10),  # Set `pretrained=True` to use the pre-trained weights
    'resnet18': ResNet18(num_classes=10)
}


# load_dataset = {
#     'CIFAR10': load_torchvision_data('CIFAR10', resize=28, maxsize=2000)[0],
#     'MNIST': load_torchvision_data('MNIST', resize=28, to3channels=True, maxsize=2000)[0],
#     'SVHN': load_torchvision_data('SVHN', )
    
# }

# print("CUDA is available:", torch.cuda.is_available())

source_data = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=64, to3channels=True, Label_Initialize = src_size, dataloader_or_not = True, maxsize=2000)
source_data2 = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=64, to3channels=True, Label_Initialize = src_size + main_args.batch_size, dataloader_or_not = True, maxsize=2000)

Labeled = source_data[0]['Labeled']
Labeled2 = source_data2[0]['Labeled']
Unlabeled = source_data[0]['Unlabeled']
validation = source_data[0]['valid'] #fix validation dataset
test = source_data[1]['test']     

print('Dataset: {} Acquisition: {}'.format(src_dataset, main_args.acquisition))

# Embed using a pretrained (+frozen) resnet
# embedder = resnet18(pretrained=True).eval()


def calc_OT(dataloader1, embedder, verbose = 0):
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False
    Labeled = dataloader1['Labeled']
    valid = dataloader1['valid']
    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                           src_dim = Feature_Cost_dim[src_dataset],
                           tgt_embedding = embedder,
                           tgt_dim = Feature_Cost_dim[src_dataset],
                           p = 2,
                           device=main_args.device)

    dist = DatasetDistance(Labeled, valid,
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          feature_cost = feature_cost,
                          sqrt_method = 'spectral',
                          sqrt_niters=10,
                          precision='single',
                          p = 2, entreg = 1e-1,
                          device=main_args.device)
    d = dist.distance(maxsamples = 1000)
    if verbose:
        print(f'OTDD(Labeled,Validation)={d:8.2f}')
    return d

# calc_OT(source_data[0], embedder = resnet18(pretrained=True).eval())


def get_acc_dataloader(dataloader, model, verbose = 1, validation_randomized = True):    
    args = {'n_epoch':50, 'lr':float(0.001), 'batch_size':20, 'max_accuracy':0.70, 'optimizer':'adam'} 
    dt = data_train(dataloader[0]['Labeled'].dataset, model, args)

    # Get the test accuracy of the initial model  on validation dataset

    if validation_randomized:
        valid = dataloader[0]['valid']  # need to chagne to a randomized validation set during pretrain      main phase 5000 fixed validation set
    else:
        valid = validation
    # Retrain the model and update the strategy with the result
    model = dt.train()
    # strategy.update_model(model)

    acc = dt.get_acc_on_set(valid) 

    if verbose:
        print('Initial Testing accuracy:', round(acc*100, 2), flush=True)
    return round(acc*100, 2)

def utility_sample(dataloader = source_data):
    '''Collect One Utility Sample'''
    acc = get_acc_dataloader(dataloader, model = load_data_dict[main_args.model])
    if main_args.OT_distance:
        OT_distance = calc_OT(dataloader[0], embedder = resnet18(pretrained=True).eval())
        return OT_distance, acc
    else:
        return acc

def sample_utility_samples(sample_size = main_args.sample_size):
    results = []
    
    for _ in range(sample_size):
        if _ % 10 == 0:
            print('Samples Collected {}'.format(_))
        sample_size = random.sample(range(src_size, src_size + main_args.total_rounds * main_args.batch_size),1)[0]   #sampling from source dataset and total dataset
        source_data = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=sample_size, to3channels=True, Label_Initialize = sample_size, dataloader_or_not = True, maxsize=5000)

        Labeled = source_data[0]['Labeled']
        # Unlabeled = source_data[0]['Unlabeled']
        # validation = source_data[0]['valid']
        
        if main_args.OT_distance:
            ot, acc = utility_sample(dataloader = source_data)
            print('OT Distance: {}, Accuracy: {}'.format(ot, acc))
        
            results.append([Labeled, ot, acc])
        else:
            acc = utility_sample(dataloader = source_data)
            print('Accuracy: {}'.format(acc))
        
            results.append([Labeled, acc])
        
    return results
    

# # open a file to write the pickled list
# with open('Samples_{}_Dataset_{}.pkl'.format(main_args.sample_size, main_args.dataset), 'wb') as f:
#     # use pickle.dump to pickle the list
#     pickle.dump(results, f)



# # open a file to load the pickled list
# with open('Samples_{}_Dataset_{}.pkl'.format(main_args.sample_size, main_args.dataset), 'rb') as f:
#     # use pickle.dump to pickle the list
#     results = pickle.load(f)

   
def deepset_ot(samples, Epochs = 150):
    logging.basicConfig(filename='deepset_ot.log', level=logging.INFO)
    # model = DeepSet_OT(in_features=in_dims[main_args.dataset])
    model = SetTransformer_OT(dim_input=in_dims[main_args.dataset])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # writer = SummaryWriter('runs/experiment_1')
    for epoch in range(Epochs):
        train_loss = 0

        for dataloader, ot, accuracy in samples:
            opt_transport_tensor = torch.tensor([ot], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for images, labels in dataloader:
            # Forward pass
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1)  
                outputs = model(images, opt_transport_tensor)

            # Compute loss
                loss = criterion(outputs, accuracy_tensor)

            # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        train_loss /= len(samples)
        if epoch % 10 == 0:
            logging.info('Epoch {} loss {}'.format(epoch, train_loss)) # Logging instead of print
            # writer.add_scalar('training loss', loss.item(), epoch * len(samples))
            # writer.add_scalar('accuracy', accuracy, epoch * len(samples))
            
        # if (epoch+1) % 10 == 0:
        #     writer.add_scalar('training loss', loss.item(), epoch * len(samples) + i)
        #     writer.add_scalar('accuracy', accuracy, epoch * len(samples) + i)
    torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}.pth'.format(main_args.dataset, main_args.sample_size))  
    

# def evaluate():
#     '''evaluate new utility samples calculate MSE using pretrained models'''
#     # Suppose your model is called 'model'
#     model = DeepSet_OT(in_features=in_dims[main_args.dataset])
#     model.load_state_dict(torch.load('Net_{}_Sample_Size_{}.pth'.format(main_args.dataset, 100)))
#     model.eval() # Set the model to evaluation mode

#     results = sample_utility_samples(sample_size = main_args.sample_size)
#     criterion = nn.MSELoss()
#     test_loss = 0
#     for one_dataloader, ot, accuracy in results:
#             opt_transport_tensor = torch.tensor([ot], device=main_args.device)
#             accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
#             for images, labels in one_dataloader:
#                 if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
#                     images = images.mean(dim=1)
#                     images = images.view(images.size(0), -1) 
#                 outputs = model(images, opt_transport_tensor).to(device=main_args.device)

#             # Compute loss
#                 loss = criterion(outputs, accuracy_tensor)
#                 print('Predicted Value: {}, True Value:{}'.format(outputs.detach().cpu().numpy(), accuracy_tensor.detach().cpu().numpy()))
#                 test_loss += loss.item()
#     test_loss /= len(results)
#     print('Test Loss is {}'.format(test_loss))
#     with open('Loss_Evaluate.txt', 'w') as file:
#         file.write(str(test_loss))
#     return test_loss
           


def evaluate():
    '''evaluate new utility samples calculate MSE'''
    criterion = nn.MSELoss()
    test_loss = 0
    if main_args.OT_distance:
        os.makedirs('{}/OT/Net_Trained_{}_Samples_{}'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size), exist_ok = True)

        model = DeepSet_OT(in_features=in_dims[main_args.dataset])
        model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet_OT.pth'.format(main_args.dataset, main_args.Net_trained)))
        model.eval() # Set the model to evaluation mode
        
        utility_samples = sample_utility_samples(sample_size = main_args.sample_size)
        for dataloader, ot, accuracy in utility_samples:
            opt_transport_tensor = torch.tensor([ot], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for images, labels in dataloader:
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = model(images, opt_transport_tensor).to(device=main_args.device)

            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                test_loss += loss.item()
        test_loss /= len(utility_samples)
        print('Test Loss is {}'.format(test_loss))
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        with open('{}/OT/Net_Trained_{}_Samples_{}/Loss_Evaluate_OT_Net_Trained_on_{}_{}_Time{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size, main_args.Net_trained, main_args.sample_size, timestamp), 'w') as file:
            file.write(str(test_loss))
        return test_loss
    else:
        os.makedirs('{}/NonOT/Net_Trained_{}_Samples_{}'.format(main_args.dataset,main_args.Net_trained, main_args.sample_size), exist_ok = True)

        model = DeepSet(in_features=in_dims[main_args.dataset])
        model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, main_args.Net_trained)))
        model.eval() # Set the model to evaluation mode
        
        utility_samples = sample_utility_samples(sample_size = main_args.sample_size)
        for dataloader, accuracy in utility_samples:
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for images, labels in dataloader:
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = model(images).to(device=main_args.device)

            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                test_loss += loss.item()
        test_loss /= len(utility_samples)
        print('Test Loss is {}'.format(test_loss))
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        with open('{}/NonOT/Net_Trained_{}_Samples_{}/Loss_Evaluate_NonOT_Net_Trained_on_{}_{}_Time{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size, main_args.Net_trained, main_args.sample_size, timestamp), 'w') as file:
            file.write(str(test_loss))
        return test_loss
    
   

# results = sample_utility_samples() 
# deepset_ot(results, Epochs = 150)     

evaluate()


    
    
        
        





