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
from torchvision.models import resnet18, vgg16
from torch.optim.lr_scheduler import ReduceLROnPlateau





import argparse
from argparse import ArgumentParser

import pickle
import datetime

from NN_classification import SetTransformer_OT, DeepSet_OT, DeepSet, OT_Net, DeepSet_Sigmoid, DeepSet_cifar
from torch.utils.tensorboard import SummaryWriter

import sys
# sys.path.append('distil/')
import distil
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, CoreSet, BatchBALDDropout, LeastConfidenceSampling

# from distil.active_learning_strategies.random_sampling import RandomSampling   # All active learning strategies showcased in this example
from distil.utils.models.resnet import ResNet18                                                 # The model used in our image classification example
from distil.utils.train_helper import data_train      # A utility training class provided by DISTIL



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=15, choices = [5, 10, 15, 20])  
parser.add_argument('--total_rounds', type=int, default=30)
parser.add_argument('--Label_Initialize', type=int, default = 20, choices = [5, 10, 20, 30, 40])
parser.add_argument('--model', type=str, default='resnet18', choices = ['vgg16', 'resnet18'])
parser.add_argument('--dataset', type=str, default='MNIST', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--num_repeats', type=int, default=10)
parser.add_argument('--acquisition', type=str, default='BADGE', choices=['random', 'GLISTER', 'CoreSet', 'BADGE'])
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
parser.add_argument('--sample_size', type=int, default=5, choices=[5, 10, 20, 30, 100, 50, 80, 40, 120]) # #of utility samples collected during pretraining
parser.add_argument('--OT_distance', type=int, default=1, choices=[1, 0])
parser.add_argument('--OT_distance_only', type=int, default=1, choices=[1, 0])
parser.add_argument('--Epochs', type=int, default=500, choices=[300, 400, 500, 600, 700, 800, 1000])
parser.add_argument('--Sigmoid', type=int, default=0, choices=[0, 1])   #whether project into values between [0,1]     1 = True

main_args = parser.parse_args()

## MNIST: Epochs = 700


## Below are for Pretraining network

#MNIST might not have OT_distance higher accuracy lower
#CIFAR10 has better prediction
DATASET_SIZES = {
    'MNIST': 28,
    'SVHN': 32,
    'CIFAR10': 28,
    'USPS': 28
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
    'SVHN': (3, 32, 32),
    'USPS': (3, 32, 32)       
}

in_dims = {
    'MNIST': int(28*28),
    'CIFAR10': int(28*28),
    'SVHN': int(32*32),    #apply mean before inputing to deepsets model
    'USPS': int(32*32)
}

# to3channel = {
#     'MNIST': True,
#     'CIFAR10': True,
#     'SVHN': True,    #apply mean before inputing to deepsets model
#     'USPS': False
# }

dtst = main_args.dataset

src_size = main_args.Label_Initialize

# Load MNIST/CIFAR in 3channels (needed by torchvision models)
src_dataset = main_args.dataset
src_dataset2 = 'SVHN'
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

source_data2 = load_torchvision_data_active_learn(src_dataset, resize=resize, to3channels=True, Label_Initialize = src_size, dataloader_or_not = True, maxsize=500, batch_size=64)  #batch_size=64,
# source_data2 = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=64, to3channels=True, Label_Initialize = src_size + main_args.batch_size, dataloader_or_not = True, maxsize=2000)

Labeled = source_data2[0]['Labeled']
# Labeled2 = source_data2[0]['Labeled']
Unlabeled = source_data2[0]['Unlabeled']
validation = source_data2[0]['valid'] #fix validation dataset it will be used over the whole script
# test = source_data[1]['test']     

print('Dataset: {} Acquisition: {}'.format(src_dataset, main_args.acquisition))

# Embed using a pretrained (+frozen) resnet
# embedder = resnet18(pretrained=True).eval()


def calc_OT(dataloader1, embedder, verbose = 0):
    '''calculate Optimal Transport distance with Feature Cost'''
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False
    Labeled = dataloader1['Labeled']
    # valid = dataloader1['valid']
    valid = validation
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


def get_acc_dataloader(dataloader, model, verbose = 1, validation_randomized = True, sigmoid = main_args.Sigmoid):    
    '''Get accuracy on randomized/non_randomized validation set'''
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
    if not sigmoid:
        return round(acc*100, 2)
    else:
        return acc

# get_acc_dataloader(source_data, model=ResNet18(num_classes=10))

def utility_sample(dataloader, sigmoid = main_args.Sigmoid):
    '''Collect One Utility Sample'''
    acc = get_acc_dataloader(dataloader, model = load_data_dict[main_args.model], sigmoid = sigmoid)
    if main_args.OT_distance:
        OT_distance = calc_OT(dataloader[0], embedder = resnet18(pretrained=True).eval())
        return OT_distance, acc
    else:
        return acc
    







def sample_utility_samples(sample_size = main_args.sample_size, ot_distance_only = False):
    results = []
    
    for _ in range(sample_size):
        if _ % 10 == 0:
            print('Samples Collected {}'.format(_))
        sample_size = random.sample(range(src_size, src_size + main_args.total_rounds * main_args.batch_size),1)[0]   #sampling from source dataset and total dataset
        source_data_inner = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=sample_size, to3channels=True, Label_Initialize = sample_size, dataloader_or_not = True, maxsize=500)

        Labeled = source_data_inner[0]['Labeled']
       
        
        
        if main_args.OT_distance:
            ot, acc = utility_sample(dataloader = source_data_inner)
            print('OT Distance: {}, Accuracy: {}'.format(ot, acc))
            if ot_distance_only:
                results.append([ot, acc])
            else:
                results.append([Labeled, ot, acc])
        else:
            acc = utility_sample(dataloader = source_data_inner)
            print('Accuracy: {}'.format(acc))
        
            results.append([Labeled, acc])
        
    return results   

### Need to replace by OT_distance3.py
def evaluate():
    '''evaluate new utility samples calculate MSE'''
    criterion = nn.MSELoss()
    test_loss = 0
    if main_args.OT_distance:
        net = torch.load('Net_{}_Sample_Size_{}.pth'.format(main_args.dataset, 80))
        utility_samples = sample_utility_samples(sample_size = main_args.sample_size)
        for dataloader, ot, accuracy in utility_samples:
            opt_transport_tensor = torch.tensor([ot], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for images, labels in dataloader:
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = net(images, opt_transport_tensor).to(device=main_args.device)

            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                test_loss += loss.item()
        test_loss /= len(utility_samples)
        print('Test Loss is {}'.format(test_loss))
        with open('Loss_Evaluate_OT_Net_Trained_on_{}.txt'.format(80), 'w') as file:
            file.write(test_loss)
        return test_loss
    elif not main_args.OT_distance_only:
        net = torch.load('Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, 80))
        utility_samples = sample_utility_samples(sample_size = main_args.sample_size)
        for dataloader, accuracy in utility_samples:
            opt_transport_tensor = torch.tensor([ot], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for images, labels in dataloader:
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = net(images, opt_transport_tensor).to(device=main_args.device)

            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                test_loss += loss.item()
        test_loss /= len(utility_samples)
        print('Test Loss is {}'.format(test_loss))
        with open('Loss_Evaluate_Net_Trained_on_{}.txt'.format(80), 'w') as file:
            file.write(test_loss)
        return test_loss
    
    
 
    
    
    
def deepset_ot(samples, Epochs = 150, tolerance = 1):
    model = DeepSet_OT(in_features=in_dims[main_args.dataset])
    # model = SetTransformer_OT(dim_input=in_dims[main_args.dataset])
    criterion = nn.MSELoss()
    if main_args.dataset == 'MNIST':
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    writer = SummaryWriter('runs/DeepSet_OT')
    print('DeepSet + OT')
    true_values = []
    predicted_values = []
    
    
    for epoch in range(Epochs):
        train_loss = 0

        for dataloader, ot, accuracy in samples:
            opt_transport_tensor = torch.tensor([ot], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            true_values.append(accuracy)
            for images, labels in dataloader:
            # Forward pass
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = model(images, opt_transport_tensor)

                
            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                predicted_values.append(outputs.item())

            # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        train_loss /= len(samples)
        if epoch % 10 == 0:
            print('Epoch {} loss {}'.format(epoch, train_loss))
        if train_loss <= tolerance:
            break
        writer.add_scalar('training loss', train_loss, epoch)
        # writer.add_scalar('accuracy', accuracy, epoch)
    writer.close()
    torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_DeepSet_OT.pth'.format(main_args.dataset, main_args.sample_size))  
    
    with open('Training_deepset_ot_predicted_vs_true_{}_{}.txt'.format(main_args.dataset, main_args.sample_size), 'w') as file:
        for true, predicted in zip(true_values, predicted_values):
            file.write(f"{true}, {predicted}\n")
    return
    

def deepset(samples, Epochs = 150, tolerance = 1):
    '''ablation study: without OT'''
    if main_args.Sigmoid:
        model = DeepSet_Sigmoid(in_features=in_dims[main_args.dataset])
    else:
        model = DeepSet(in_features=in_dims[main_args.dataset])
    
    # model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, 100)))
    # model.eval()
    criterion = nn.MSELoss()
    
    if main_args.dataset == 'SVHN' and main_args.sample_size == 20:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    elif main_args.dataset == 'SVHN' and main_args.sample_size == 50:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    elif main_args.dataset == 'CIFAR10' and main_args.sample_size == 20:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    writer = SummaryWriter('runs/DeepSet_only')
    print('Ablation Study deepset only')
    
    true_values = []
    predicted_values = []
    for epoch in range(Epochs):
        train_loss = 0

        for dataloader, accuracy in samples:
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            true_values.append(accuracy)
            for images, labels in dataloader:
            # Forward pass
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = model(images)
                
                predicted_values.append(outputs.item())
            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
            # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  

                optimizer.step()
                train_loss += loss.item()
        train_loss /= len(samples)
        scheduler.step(train_loss)
        
        if train_loss <= tolerance:
            print('Training Loss: ', train_loss)
            break
        if epoch % 10 == 0:
            print('Epoch {} loss {}'.format(epoch, train_loss))
        writer.add_scalar('training loss', train_loss, epoch)
        # writer.add_scalar('accuracy', accuracy, epoch)
    if not main_args.Sigmoid:
        torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, main_args.sample_size))  
    else:
        torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_DeepSet_Sigmoid.pth'.format(main_args.dataset, main_args.sample_size))  
    writer.close()
    with open('Training_deepset_predicted_vs_true_{}_{}.txt'.format(main_args.dataset, main_args.sample_size), 'w') as file:
        for true, predicted in zip(true_values, predicted_values):
            file.write(f"{true}, {predicted}\n")
    return
    

def ot(samples, Epochs = 200, tolerance = 1):
    '''ablation study: with OT and without data'''
    model = OT_Net(input_size = 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    writer = SummaryWriter('runs/OT_only')
    print('Ablation Study OT only')
    true_values = []
    predicted_values = []
    for epoch in range(Epochs):
        train_loss = 0

        for ot_distance, accuracy in samples:
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            
            outputs = model(torch.tensor([[ot_distance]], device=main_args.device))

            true_values.append(accuracy)
            # Compute loss
            loss = criterion(outputs, accuracy_tensor)
            predicted_values.append(outputs.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(samples)
        if train_loss <= tolerance:
            break
        if epoch % 10 == 0:
            print('Epoch {} loss {}'.format(epoch, train_loss))
        writer.add_scalar('training loss', train_loss, epoch)
        # writer.add_scalar('accuracy', accuracy, epoch)
    writer.close()
    torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_OT_only.pth'.format(main_args.dataset, main_args.sample_size))  
    with open('Training_ot_predicted_vs_true_{}_{}.txt'.format(main_args.dataset, main_args.sample_size), 'w') as file:
        for true, predicted in zip(true_values, predicted_values):
            file.write(f"{true}, {predicted}\n")
    
    return


# def three_comparisons():
#     results = sample_utility_samples()   #ot_distance = 1 ot_distance_only = 0
#     ot([result[1:] for result in results], Epochs = main_args.Epochs) #excluding labeled
#     deepset_ot(results, Epochs = main_args.Epochs)
#     deepset([[result[0], results[2]] for result in results], Epochs = main_args.Epochs) #exclude ot
    

if main_args.OT_distance_only:  #ablation study
    results = sample_utility_samples(ot_distance_only=True)
    ot(results, Epochs = main_args.Epochs)
elif main_args.OT_distance:
    results = sample_utility_samples()
    deepset_ot(results, Epochs = main_args.Epochs)
else: #ablation study
    results = sample_utility_samples()
    deepset(results, Epochs = main_args.Epochs)
    

    
    
    
##### TODO Main Algorithm we want to write: Two Stage Utility Learning #######
def calc_OT_interpolate(dataloader1, dataloader2, embedder, verbose = 0):
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False
    Labeled = dataloader1['Labeled']
    Labeled2 = dataloader2['Labeled']
    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                           src_dim = Feature_Cost_dim[src_dataset],
                           tgt_embedding = embedder,
                           tgt_dim = Feature_Cost_dim[src_dataset],
                           p = 2,
                           device=main_args.device)

    dist = DatasetDistance(Labeled, Labeled2,
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

def interpolate_utility_value(dataloader1, dataloader2, dataloader_between, acc1, acc2):
    d2b = calc_OT_interpolate(dataloader_between, dataloader2['Labeled']) 
    d1b = calc_OT_interpolate(dataloader1['Labeled'], dataloader_between)
    return (d2b * acc1 + d1b * acc2)/(d2b + d1b)



def concat_dataloader(dataloader1, dataloader2):
# Suppose you have dataloaders `dataloader1` and `dataloader2`
    dataset1 = dataloader1['Labeled'].dataset
    dataset2 = dataloader2['Labeled'].dataset

    # Concatenate datasets
    combined_dataset = ConcatDataset([dataset1, dataset2])
    
    total_size = len(combined_dataset)

    # Create new dataloader
    combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=total_size)  # set batch size as needed


    num_samples = random.sample(range(len(combined_dataset)), 1)[0]  #generate a random length sample
    indices = random.sample(range(len(combined_dataset)), num_samples)


# Now, we create a subset.
    subset = Subset(combined_dataset, indices)
    
    subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset))
    
    return subset_dataloader





# # results = sample_utility_samples()
# loss = evaluate()

# print('Final Test Loss is', loss)




# evaluate(results)

# # open a file to write the pickled list
# with open('Samples_{}_Dataset_{}.pkl'.format(main_args.sample_size, main_args.dataset), 'wb') as f:
#     # use pickle.dump to pickle the list
#     pickle.dump(results, f)



# open a file to load the pickled list
# with open('Samples_{}_Dataset_{}.pkl'.format(main_args.sample_size, main_args.dataset), 'rb') as f:
#     # use pickle.dump to pickle the list
#     results = pickle.load(f)

# def generate_utility_samples():




    # for i in range(len(samples)):
        
    #     for epoch in range(Epochs):
    #         for images, labels in dataloader:
    #     # Forward pass
    #             outputs = model(images)

    #     # Compute loss
    #     # Here we assume that "accuracy" is a PyTorch tensor containing the accuracy of the model on the current batch
    #             accuracy = torch.FloatTensor(samples[2])
    #             loss = criterion(outputs, accuracy)

    #     # Backward pass and optimization
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    
    
        
        



# # Here we use same embedder for both datasets
# feature_cost = FeatureCost(src_embedding = embedder,
#                            src_dim = Feature_Cost_dim[src_dataset],
#                            tgt_embedding = embedder,
#                            tgt_dim = Feature_Cost_dim[src_dataset],
#                            p = 2,
#                            device='cpu')

# dist = DatasetDistance(Labeled, Labeled2,
#                           inner_ot_method = 'exact',
#                           debiased_loss = True,
#                           feature_cost = feature_cost,
#                           sqrt_method = 'spectral',
#                           sqrt_niters=10,
#                           precision='single',
#                           p = 2, entreg = 1e-1,
#                           device='cpu')


# # # Instantiate distance
# # dist = DatasetDistance(Labeled, Labeled2,
# #                           inner_ot_method = 'exact',
# #                           debiased_loss = True,
# #                           p = 2, entreg = 1e-1,
# #                           device='cpu')

# d = dist.distance(maxsamples = 1000)
# print(f'OTDD(Labeled,Labeled2)={d:8.2f}')


# network = SetTransformer_OT(dim_input = in_dims[main_args.dataset])




# def train_one(acquisition_type = 'BADGE', batch_size = main_args.batch_size, Labeled = Labeled, Unlabeled = Unlabeled, test = test, model = load_data_dict[main_args.model], n_class = num_classes, n_rounds = main_args.total_rounds, repeat = False, trial_num = 0):
    
#     if acquisition_type == 'random':
#         strategy_args = {'batch_size' : batch_size, 'device' : 'cuda'}  #Budget per round
#         strategy = RandomSampling(Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)

#     elif acquisition_type == 'GLISTER':
#         strategy_args = {'lr':0.05}
#         strategy = GLISTER( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)
   
#     elif acquisition_type == 'CoreSet':
#         strategy_args = {}
#         strategy = CoreSet( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)

#     elif acquisition_type == 'BADGE':
#         strategy_args = {}
#         strategy = BADGE( Labeled, LabeledToUnlabeledDataset(Unlabeled), net=model, nclasses=n_class, args=strategy_args)
    

#     # Use the same training parameters as before
#     args = {'n_epoch':100, 'lr':float(0.001), 'batch_size':20, 'max_accuracy':0.99, 'optimizer':'adam'} 
#     dt = data_train(Labeled, model, args)

#     # Update the model used in the AL strategy with the loaded initial model
#     strategy.update_model(model)

#     # Get the test accuracy of the initial model
#     acc = np.zeros(n_rounds)
#     acc[0] = dt.get_acc_on_set(test)
#     print('Initial Testing accuracy:', round(acc[0]*100, 2), flush=True)

#     # User Controlled Loop
#     for rd in range(1, n_rounds):
#         print('-------------------------------------------------')
#         print('Round', rd) 
#         print('-------------------------------------------------')

#     # Use select() to obtain the indices in the unlabeled set that should be labeled
#         # organ_amnist_full_train.transform = organ_amnist_test_transform       # Disable augmentation while selecting new points as to not interfere with the strategies
#         idx = strategy.select(batch_size)
#         # organ_amnist_full_train.transform = organ_amnist_training_transform   # Enable augmentation

#     # Add the selected points to the train set. The unlabeled set shown in the next couple lines 
#     # already has the associated labels, so no human labeling is needed. Again, this is because 
#     # we already have the labels a priori. In real scenarios, a human oracle would need to provide 
#     # then before proceeding.
#         Labeled = ConcatDataset([Labeled, Subset(Unlabeled, idx)])
#         Remaining_unlabeled_idx = list(set(range(len(Unlabeled))) - set(idx))
#         Unlabeled = Subset(Unlabeled, Remaining_unlabeled_idx)

#         print('Number of Labeled points -', len(Labeled))

#         # Update the data used in the AL strategy and the training class
#         strategy.update_data(Labeled, LabeledToUnlabeledDataset(Unlabeled))
#         dt.update_data(Labeled)

#         # Retrain the model and update the strategy with the result
#         model = dt.train()
#         strategy.update_model(model)

#         # Get new test accuracy
#         acc[rd] = dt.get_acc_on_set(test)
#         print('Testing accuracy:', round(acc[rd]*100, 2), flush=True)

#     print('Training Completed')

#     if repeat == True:
#         now = datetime.datetime.now()
#         timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
#         file_name = "Accuracy_For_{}_Total_Rounds_{}_For_Dataset_{}_Time_{}.txt".format(acquisition_type, main_args.total_rounds, main_args.dataset, timestamp)
#         with open(os.path.join(base_dir,file_name), 'w') as f:
#             for item in acc:
#                 f.write("%s\n" % item)
#         return acc
        
#     # Lastly, we save the accuracies in case a comparison is warranted.
#     with open(os.path.join(base_dir,'Acquisition Type {} for Dataset {}.txt'.format(acquisition_type, main_args.dataset)), 'w') as f:
#         for item in acc:
#             f.write("%s\n" % item)
#     return acc


# def training_loop(acquisition_type = 'CoreSet', trials = main_args.num_repeats):
#     trials_acc = []
#     for trial in range(trials):
#         one_trial_acc = train_one(acquisition_type=acquisition_type, repeat = True)
#         trials_acc.append(one_trial_acc)
#         print('Trial {}: Accuracy {}'.format(trial+1, one_trial_acc))
#         # # Get the current date and time
#         # now = datetime.datetime.now()

#         # # Format the timestamp as a string (e.g., '2023-05-05_15-30-45')
#         # timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

#         # # Concatenate the timestamp with the file name
#         # file_name = "model_{}_For_{}_Trial_{}.pt".format(timestamp, acquisition_type, trial+1)

#         # # Save the model
#         # torch.save(model.state_dict(), file_name)
#     with open(os.path.join(base_dir, 'Acquisition Type {} for Dataset {} Trials {}.pkl'.format(acquisition_type, main_args.dataset, trials)), 'wb') as fp:
#         pickle.dump(trials_acc, fp)
#     return trials_acc
        
            
# #train_one(acquisition_type='BatchBALD')       
# # training_loop(acquisition_type='CoreSet')











