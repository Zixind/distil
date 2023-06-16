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

from NN_classification import SetTransformer_OT, DeepSet_OT, DeepSet, OT_Net, DeepSet_Sigmoid
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
parser.add_argument('--Label_Initialize', type=int, default = 20, choices = [20, 30, 40])
parser.add_argument('--model', type=str, default='resnet18', choices = ['vgg16', 'resnet18'])
parser.add_argument('--dataset', type=str, default='MNIST', choices = ['SVHN', 'MNIST', 'CIFAR10', 'USPS'])
parser.add_argument('--num_repeats', type=int, default=10)
parser.add_argument('--acquisition', type=str, default='BADGE', choices=['random', 'GLISTER', 'CoreSet', 'BADGE'])
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
parser.add_argument('--OT_distance', type=int, default=1, choices=[1, 0])
parser.add_argument('--Net_trained', type=int, default=80, choices=[10,20,50,80,100])
parser.add_argument('--OT_distance_only', type=int, default=1, choices=[1, 0])
parser.add_argument('--sample_size', type=int, default=5, choices=[5, 10, 20, 30, 100, 50, 80, 70, 120])
parser.add_argument('--Sigmoid', type=int, default=0, choices=[0, 1])   #whether project into values between [0,1]     1 = True

main_args = parser.parse_args()

#For resize in load_torchvision_data_active_learn
DATASET_SIZES = {
    'MNIST': 28,
    'SVHN': 32,   #original 32
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
    'USPS': (3, 28, 28)       
}

# in_dims = {
#     'MNIST': int(28*28),
#     'CIFAR10': int(28*28),
#     'SVHN': int(32*32)
# }

in_dims = {
    'MNIST': int(3*28*28),
    'CIFAR10': int(3*28*28),   #original 32*32
    'SVHN': int(3*32*32),    #apply mean before inputing to deepsets model
    'USPS': int(3*28*28)
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

Labeled = source_data[0]['Labeled']
Unlabeled = source_data[0]['Unlabeled']
validation = source_data[0]['valid'] #fix validation dataset
test = source_data[1]['test']     

print('Dataset: {} Acquisition: {} Net Trained {}'.format(src_dataset, main_args.acquisition, main_args.Net_trained))

# Embed using a pretrained (+frozen) resnet
# embedder = resnet18(pretrained=True).eval()


def calc_OT(dataloader1, embedder, verbose = 0):
    '''calculate Optimal Transport distance with Feature Cost'''
    '''dataloder1 is dataloader[0]'''
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False
    Labeled = dataloader1['Labeled']
    valid = dataloader1['valid']
    
    
    #convert Dataset to Dataloader
    print('Labeled Length: {}, Valid Length: {}'.format(len(Labeled), len(valid)))
    Labeled_dataloader = DataLoader(Labeled, batch_size=len(Labeled), shuffle=True) 
    valid_dataloader = DataLoader(valid, batch_size=len(valid), shuffle=True) 
    
    
    # valid = validation
    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                           src_dim = Feature_Cost_dim[src_dataset],
                           tgt_embedding = embedder,
                           tgt_dim = Feature_Cost_dim[src_dataset],
                           p = 2,
                           device=main_args.device)

    dist = DatasetDistance(Labeled_dataloader, valid_dataloader,
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          feature_cost = feature_cost,
                          sqrt_method = 'spectral',
                          sqrt_niters=10,
                          precision='single',
                          p = 2, entreg = 1e-1,
                          device=main_args.device, verbose = 0)
    d = dist.distance(maxsamples = 1000)
    if verbose:
        print(f'OTDD(Labeled,Validation)={d:8.2f}')
    return d


def get_acc_dataloader(dataloader, model, verbose = 1, validation_randomized = True, sigmoid = main_args.Sigmoid):    
    args = {'n_epoch':100, 'lr':float(0.001), 'batch_size':50, 'max_accuracy':0.90, 'optimizer':'adam', 'isverbose':1, 'islogs':1} 
    dt = data_train(dataloader[0]['Labeled'].dataset, model, args)
    
    #dataloader[0]['Labeled'] is a dataset

    # Get the test accuracy of the initial model  on validation dataset

    if validation_randomized:
        valid = dataloader[0]['valid']  # need to chagne to a randomized validation set during pretrain      main phase 5000 fixed validation set
   
    # Retrain the model and update the strategy with the result
    model = dt.train()
    # strategy.update_model(model)

    acc = dt.get_acc_on_set(valid) 

    if verbose:
        print('Initial Validation accuracy:', round(acc*100, 2), flush=True)
    if not sigmoid:
        return round(acc*100, 2)
    else:
        return acc

def utility_sample(dataloader, sigmoid = main_args.Sigmoid):
    '''Collect One Utility Sample'''
    acc = get_acc_dataloader(dataloader, model = load_data_dict[main_args.model], sigmoid = sigmoid)
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
        if main_args.dataset in ['MNIST', 'USPS']:
            source_data_inner = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=sample_size, to3channels=True, Label_Initialize = sample_size, dataloader_or_not = True, maxsize=500)
        else:
            source_data_inner = load_torchvision_data_active_learn(src_dataset, resize=resize, batch_size=sample_size, to3channels=False, Label_Initialize = sample_size, dataloader_or_not = True, maxsize=500)

        Labeled = source_data_inner[0]['Labeled']
        
        Labeled_dataloader = DataLoader(Labeled, batch_size=len(Labeled), shuffle=True)   #make the dataloader only contains one batch(one single utility sample)

        if main_args.OT_distance_only:
            ot, acc = utility_sample(dataloader = source_data_inner)
            print('OT Distance: {}, Accuracy: {}'.format(ot, acc))
            results.append([ot, acc])
        elif main_args.OT_distance:
            ot, acc = utility_sample(dataloader = source_data_inner)
            print('OT Distance: {}, Accuracy: {}'.format(ot, acc))
        
            results.append([Labeled_dataloader, ot, acc])
        else:
            acc = utility_sample(dataloader = source_data_inner)
            print('Accuracy: {}'.format(acc))
        
            results.append([Labeled_dataloader, acc])
        
    return results
    
   
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
    
           
def deepset(samples, Epochs = 150, tolerance  = 1):
    '''ablation study: without OT'''
    if main_args.Sigmoid:
        model = DeepSet_Sigmoid(in_features=in_dims[main_args.dataset])
    else:
        model = DeepSet(in_features=in_dims[main_args.dataset])
    
    # model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, 100)))
    # model.eval()
    criterion = nn.MSELoss()
    
    if main_args.dataset == 'SVHN':
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    writer = SummaryWriter('runs/DeepSet_only')
    print('Ablation Study deepset only')
    
    for epoch in range(Epochs):
        train_loss = 0

        for dataloader, accuracy in samples:
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            
            for images, labels in dataloader:
            # Forward pass
                if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                    images = images.mean(dim=1)
                    images = images.view(images.size(0), -1) 
                    # print(images.shape) 
                outputs = model(images)
            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
            # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        train_loss /= len(samples)
        if train_loss <= tolerance:
            print('Training Loss: ', train_loss)
            break
        if epoch % 10 == 0:
            print('Epoch {} loss {}'.format(epoch, train_loss))
        if (epoch+1) % 10 == 0:
            writer.add_scalar('training loss', loss.item())
            writer.add_scalar('accuracy', accuracy)
    if not main_args.Sigmoid:
        torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, main_args.sample_size))  
    else:
        torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_DeepSet_Sigmoid.pth'.format(main_args.dataset, main_args.sample_size))  
    writer.close()
    return

def evaluate():
    '''evaluate new utility samples calculate MSE'''
    criterion = nn.MSELoss()
    test_loss = 0
    
    if main_args.OT_distance_only:
        os.makedirs('{}/OT_only/Net_Trained_{}_Samples_{}'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size), exist_ok = True)

        model = OT_Net(input_size = 1)
        model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_OT_only.pth'.format(main_args.dataset, main_args.Net_trained)))
        model.eval() # Set the model to evaluation mode
        
        utility_samples = sample_utility_samples()
        print('Evaluation OT only')
        for ot, accuracy in utility_samples:
            opt_transport_tensor = torch.tensor([[ot]], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for ot_distance, accuracy in utility_samples:
                accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            
                outputs = model(torch.tensor([[ot_distance]], device=main_args.device))


                # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                print('Prediction {}. True Value {}'.format(outputs, accuracy_tensor))
                test_lsoss += loss.item()
        test_loss /= len(utility_samples)
        print('OT Only Test Loss is {}'.format(test_loss))
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        with open('{}/OT_only/Net_Trained_{}_Samples_{}/Loss_Evaluate_OT_Net_Trained_on_{}_{}_Time{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size, main_args.Net_trained, main_args.sample_size, timestamp), 'w') as file:
            file.write(str(test_loss))
        return test_loss
    elif main_args.OT_distance:
        os.makedirs('{}/OT/Net_Trained_{}_Samples_{}'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size), exist_ok = True)

        model = DeepSet_OT(in_features=in_dims[main_args.dataset])
        model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet_OT.pth'.format(main_args.dataset, main_args.Net_trained)))
        model.eval() # Set the model to evaluation mode
        
        utility_samples = sample_utility_samples()
        print('Evaluation DeepSets OT')
        for dataloader, ot, accuracy in utility_samples:
            opt_transport_tensor = torch.tensor([ot], device=main_args.device)
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            for images, labels in dataloader:
            # Forward pass
                images = images.view(images.shape[0], -1)
                outputs = model(images, opt_transport_tensor).to(device=main_args.device)

            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                print('Prediction {}. True Value {}'.format(outputs, accuracy_tensor))
                test_loss += loss.item()
        test_loss /= len(utility_samples)
        print('DeepSets OT Test Loss is {}'.format(test_loss))
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        with open('{}/OT/Net_Trained_{}_Samples_{}/Loss_Evaluate_OT_Net_Trained_on_{}_{}_Time{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size, main_args.Net_trained, main_args.sample_size, timestamp), 'w') as file:
            file.write(str(test_loss))
        return test_loss
    else:
        if main_args.Sigmoid == False:
            os.makedirs('{}/NonOT/Net_Trained_{}_Samples_{}'.format(main_args.dataset,main_args.Net_trained, main_args.sample_size), exist_ok = True)

            model = DeepSet(in_features=in_dims[main_args.dataset])
            
            state_dict = torch.load('Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, main_args.Net_trained))
            # # Filter out the state_dict to only include keys that exist in the current model
            # state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
            model.load_state_dict(state_dict)
            model.eval()  # Set the model to evaluation mode
        else:
            os.makedirs('{}/NonOT_Sigmoid/Net_Trained_{}_Samples_{}'.format(main_args.dataset,main_args.Net_trained, main_args.sample_size), exist_ok = True)

            model = DeepSet_Sigmoid(in_features=in_dims[main_args.dataset])
            model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet_Sigmoid.pth'.format(main_args.dataset, main_args.Net_trained)))
            model.eval() # Set the model to evaluation mode
        
        utility_samples = sample_utility_samples()
        print('Evaluation DeepSets')
        true_values = []
        predicted_values = []
        for dataloader, accuracy in utility_samples:
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            true_values.append(accuracy)
            for images, labels in dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model(images).to(device=main_args.device)
                
            # Compute loss
                loss = criterion(outputs, accuracy_tensor)
                print('Prediction {}. True Value {}'.format(outputs, accuracy_tensor))
                test_loss += loss.item()
        test_loss /= len(utility_samples)
        print('DeepSets Test Loss is {}'.format(test_loss))
        
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        with open('{}/NonOT/Net_Trained_{}_Samples_{}/Loss_Evaluate_NonOT_Net_Trained_on_{}_{}_Time{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size, main_args.Net_trained, main_args.sample_size, timestamp), 'w') as file:
            file.write(str(test_loss))
        
        with open('Evaluation_deepset_predicted_vs_true_{}_{}_{}_{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size, timestamp), 'w') as file:
            for true, predicted in zip(true_values, predicted_values):
                file.write(f"{true}, {predicted}\n")
        return test_loss
    
def ot(samples, Epochs = 200, tolerance = 1):
    '''ablation study: with OT and without data'''
    model = OT_Net(input_size = 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    writer = SummaryWriter('runs/OT_only')
    print('Ablation Study OT only')
    for epoch in range(Epochs):
        train_loss = 0

        for ot_distance, accuracy in samples:
            accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
            
            outputs = model(torch.tensor([[ot_distance]], device=main_args.device))


            # Compute loss
            loss = criterion(outputs, accuracy_tensor)

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
        if (epoch+1) % 10 == 0:
            writer.add_scalar('training loss', loss.item())
            writer.add_scalar('accuracy', accuracy)
    torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}_OT_only.pth'.format(main_args.dataset, main_args.sample_size))  
    
    writer.close()
    return

# results = sample_utility_samples() 
# deepset_ot(results, Epochs = 150)     



#To run below: set ot_distance = 1 ot_distance_only = 0
def evaluate_three_comparisons():
    criterion = nn.MSELoss()
    test_loss = 0
    
    results = sample_utility_samples()   
    print('Three evaluations!')
    #ot
    utility_samples_ot = [result[1:] for result in results]
    
    model = OT_Net(input_size = 1)
    model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_OT_only.pth'.format(main_args.dataset, main_args.Net_trained)))
    model.eval() # Set the model to evaluation mode
    
    true_values = []
    predicted_values_ot = []
    for ot, accuracy in utility_samples_ot:
        opt_transport_tensor = torch.tensor([[ot]], device=main_args.device)
        accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
        true_values.append(accuracy)
        
        outputs = model(opt_transport_tensor)
        
        predicted_values_ot.append(outputs.item())

        loss = criterion(outputs, accuracy_tensor)
        print('Prediction {}. True Value {}'.format(outputs, accuracy_tensor))
        test_loss += loss.item()
    test_loss /= len(utility_samples_ot)
    print('OT Only Test Loss is {}'.format(test_loss))
    
    #deepset_ot
    model = DeepSet_OT(in_features=in_dims[main_args.dataset])
    model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet_OT.pth'.format(main_args.dataset, main_args.Net_trained)))
    model.eval() # Set the model to evaluation mode
        
    predicted_values_deepset_ot = []
    test_loss = 0
    for dataloader, ot, accuracy in results:
        opt_transport_tensor = torch.tensor([ot], device=main_args.device)
        accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
        for images, labels in dataloader:
            if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                images = images.mean(dim=1)
                images = images.view(images.size(0), -1) 
            outputs = model(images, opt_transport_tensor).to(device=main_args.device)
            predicted_values_deepset_ot.append(outputs.item())
            # Compute loss
            loss = criterion(outputs, accuracy_tensor)
            print('Prediction {}. True Value {}'.format(outputs, accuracy_tensor))
            test_loss += loss.item()
    test_loss /= len(results)
    print('DeepSets OT Test Loss is {}'.format(test_loss))
    
    #deepset
    model = DeepSet(in_features=in_dims[main_args.dataset])
    model.load_state_dict(torch.load('Net_{}_Sample_Size_{}_DeepSet.pth'.format(main_args.dataset, main_args.Net_trained)))
    model.eval() # Set the model to evaluation mode

    utility_samples_deepset = [[result[0], result[2]] for result in results]
    predicted_values_deepset = []
    test_loss = 0
    for dataloader, accuracy in utility_samples_deepset:
        accuracy_tensor = torch.tensor([[accuracy]], device=main_args.device)
        for images, labels in dataloader:
            if main_args.dataset == 'MNIST' or main_args.dataset == 'CIFAR10' or main_args.dataset == 'SVHN':
                images = images.mean(dim=1)
                images = images.view(images.size(0), -1) 
            outputs = model(images).to(device=main_args.device)
            predicted_values_deepset.append(outputs.item())

            # Compute loss
            loss = criterion(outputs, accuracy_tensor)
            print('Prediction {}. True Value {}'.format(outputs, accuracy_tensor))
            test_loss += loss.item()
    test_loss /= len(utility_samples_deepset)
    print('DeepSets Test Loss is {}'.format(test_loss))
    
    with open('Three_predicted_vs_true_{}_{}_{}.txt'.format(main_args.dataset, main_args.Net_trained, main_args.sample_size), 'w') as file:
            for true, predicted_deepset, predicted_deepset_ot, predicted_ot in zip(true_values, predicted_values_deepset, predicted_values_deepset_ot, predicted_values_ot):
                file.write(f"{true}, {predicted_deepset}, {predicted_deepset_ot}, {predicted_ot}\n")
    


evaluate()
    
# evaluate_three_comparisons()    
         
        
    
        
        





