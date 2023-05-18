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

import argparse
from argparse import ArgumentParser

import pickle
import datetime

from NN_classification import SetTransformer_OT, DeepSet_OT
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
parser.add_argument('--sample_size', type=int, default=5, choices=[5, 10, 20, 30, 100, 50])
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
    'SVHN': int(32*32*3)
}


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


def get_acc_dataloader(dataloader, model, verbose = 1):    
    args = {'n_epoch':200, 'lr':float(0.001), 'batch_size':20, 'max_accuracy':0.99, 'optimizer':'adam'} 
    dt = data_train(dataloader[0]['Labeled'], model, args)

    # Get the test accuracy of the initial model  on validation dataset

    acc = dt.get_acc_on_set(dataloader[0]['valid']) 

    if verbose:
        print('Initial Testing accuracy:', round(acc*100, 2), flush=True)
    return round(acc*100, 2)

# get_acc_dataloader(source_data, model=ResNet18(num_classes=10))

def utility_sample(dataloader = source_data):
    OT_distance = calc_OT(dataloader[0], embedder = resnet18(pretrained=True).eval())
    acc = get_acc_dataloader(dataloader, model = load_data_dict[main_args.model])
    return OT_distance, acc

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
        
        ot, acc = utility_sample(dataloader = source_data)
        print('OT Distance: {}, Accuracy: {}'.format(ot, acc))
        
        results.append([Labeled, ot, acc])
        
    return results
    
results = sample_utility_samples()

# # open a file to write the pickled list
# with open('Samples_{}_Dataset_{}.pkl'.format(main_args.sample_size, main_args.dataset), 'wb') as f:
#     # use pickle.dump to pickle the list
#     pickle.dump(results, f)



# open a file to load the pickled list
# with open('Samples_{}_Dataset_{}.pkl'.format(main_args.sample_size, main_args.dataset), 'rb') as f:
#     # use pickle.dump to pickle the list
#     results = pickle.load(f)

   
def deepset_ot(samples, Epochs = 150):
    model = DeepSet_OT(in_features=in_dims[main_args.dataset])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter('runs/experiment_1')
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
            print('Epoch {} loss {}'.format(epoch, train_loss))
        # if (epoch+1) % 10 == 0:
        #     writer.add_scalar('training loss', loss.item(), epoch * len(samples) + i)
        #     writer.add_scalar('accuracy', accuracy, epoch * len(samples) + i)
    torch.save(model.state_dict(), 'Net_{}_Sample_Size_{}.pth'.format(main_args.dataset, main_args.sample_size))  
    
    writer.close()

    
deepset_ot(results, Epochs = 150)            

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











