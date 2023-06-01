import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import collections
import copy
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.optimize as op 
import seaborn as sns

import pickle





class DeepSet_cifar(nn.Module):

    def __init__(self, in_features, set_features=512):
        super(DeepSet_cifar, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'




#### Set Transformer #######
#### Neural Network ############
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

# class DeepSet(nn.Module):
#     def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
#         super(DeepSet, self).__init__()
#         self.num_outputs = num_outputs
#         self.dim_output = dim_output
#         self.enc = nn.Sequential(
#                 nn.Linear(dim_input, dim_hidden),
#                 nn.ReLU(),
#                 nn.Linear(dim_hidden, dim_hidden),
#                 nn.ReLU(),
#                 nn.Linear(dim_hidden, dim_hidden),
#                 nn.ReLU(),
#                 nn.Linear(dim_hidden, dim_hidden))
#         self.dec = nn.Sequential(
#                 nn.Linear(dim_hidden, dim_hidden),
#                 nn.ReLU(),
#                 nn.Linear(dim_hidden, dim_hidden),
#                 nn.ReLU(),
#                 nn.Linear(dim_hidden, dim_hidden),
#                 nn.ReLU(),
#                 nn.Linear(dim_hidden, num_outputs*dim_output))

#     def forward(self, X):
#         X = self.enc(X).mean(-2)
#         X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
#         return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs = 1, dim_output = 1,
            num_inds=10, dim_hidden=40, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.criterion = nn.MSELoss(reduction = 'sum')
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))
        self.backbone = nn.Linear(2, 1)

    def forward(self, X, representation = False):
        if representation:
            return torch.mean(self.enc(X), dim = 1)
        else:
            return F.leaky_relu(self.backbone(torch.cat((self.dec(self.enc(X)).view(1,1), self.calculate_dpp(X.squeeze(0), distance = False)), dim = 1)))
    
    def calculate_dpp(self, input, distance = False):
        '''gram matrix of embedding x^{T}x x has dim(set_num, self.set_features)'''
        if distance:
            vol = torch.det(1/(1 + torch.cdist(input, input, p = 2))) #adopted from https://arxiv.org/pdf/1905.07697.pdf
        else:
            input = F.normalize(input)  #normalize embedding
            Gram = torch.mm(input, input.t())
            vol = torch.det(Gram)
        
        return vol.view(1, 1)

class SetTransformer_OT(nn.Module):
    def __init__(self, dim_input, num_outputs = 1, dim_output = 1,
            num_inds=10, dim_hidden=40, num_heads=4, ln=False):    #10 40 4
        super(SetTransformer_length, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss(reduction = 'sum')
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))
        self.backbone = nn.Linear(2, 1)

    def forward(self, X, ot_distance, representation = False):
        X = X.to(self.device)
        if representation:
            return torch.mean(self.enc(X), dim = 1)
        else:
            return F.leaky_relu(self.backbone(torch.cat((self.dec(self.enc(X)).view(1,1), ot_distance.view(1,1)), dim = 1)))
    
    # def calculate_dpp(self, input, distance = False):
    #     '''gram matrix of embedding x^{T}x x has dim(set_num, self.set_features)'''
    #     input = input.to(self.device)
    #     if distance:
    #         vol = torch.det(1/(1 + torch.cdist(input, input, p = 2))) #adopted from https://arxiv.org/pdf/1905.07697.pdf
    #     else:
    #         input = F.normalize(input)  #normalize embedding
    #         Gram = torch.mm(input, input.t())
    #         vol = torch.det(Gram)
        
    #     return vol.view(1, 1)
    
    # def length(self, input):
    #     input = input.to(self.device)
    #     return torch.Tensor(np.array([input.shape[1]])).view(1,1)





#copy from Si Chen
class DeepSet(nn.Module):

    def __init__(self, in_features, set_features=128):
        super(DeepSet, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        

        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input, representation = False):
        # Flatten the images into vectors
    
        x = self.feature_extractor(input)
        # x = x.sum(dim=1)
        x = x.sum(dim=0).unsqueeze(0)
        if representation:
            return x
        else:
            y = self.regressor(x)
            return y.view(1,1)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'
            
class DeepSet_OT(nn.Module):

    def __init__(self, in_features, set_features=128):
        super(DeepSet_OT, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.backbone = nn.Linear(2, 1)
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        self.add_module('2', self.backbone)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input, ot, representation = False):
        # Flatten the images into vectors
    
        x = self.feature_extractor(input)
        # x = x.sum(dim=1)
        x = x.sum(dim=0).unsqueeze(0)
        if representation:
            return x
        else:
            x = self.regressor(x)
            combined = torch.cat((x, ot.view(1,1)), dim=1)
            y = self.backbone(combined)
        return y.view(1,1)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'

class Utility_deepset(object):

    def __init__(self, in_dims, set_feature, lr=0.001):

        self.model = DeepSet(in_dims).cuda()
        
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss(reduction='sum')
        self.optim = optim.Adam(self.model.parameters(), lr)
        
    # train_data: X_syn
    # train_set: (X_feature, y_feature)
    def fit(self, train_X_syn, train_set, valid_set, n_epoch, batch_size=32):

      # trainX_syn: labeled data N*3*32*32, train_set: (data indices ([,,,]), utility value) (X_feature, y_feature)

        train_data = copy.deepcopy(train_X_syn)
        N, k = train_data.shape

        train_loss = 0.0
        X_feature, y_feature = train_set
        X_feature_test, y_feature_test = valid_set
        train_size = len(y_feature)

        for epoch in range(n_epoch):
          np.random.shuffle(train_data)
          train_loss = 0
          start_ind = 0
          for j in range(train_size//batch_size):
            start_ind = j*batch_size
            batch_X, batch_y = [], []
            for i in range(start_ind, min(start_ind+batch_size, train_size)):
              b = copy.deepcopy(train_data)
              zero_ind = np.where(X_feature[i]==0)[0]
              b[zero_ind] = np.zeros(k)
              batch_X.append( b )
              batch_y.append( [y_feature[i]] )
                
            batch_X, batch_y = torch.FloatTensor(np.stack(batch_X)).cuda(), torch.FloatTensor(batch_y).cuda()

            self.optim.zero_grad()
            y_pred = self.model(batch_X)
            loss = self.l2(y_pred, batch_y)
            loss_val = np.asscalar(loss.data.cpu().numpy())
            # print(loss_val)
            loss.backward()
            self.optim.step()
            train_loss += loss_val
          train_loss /= train_size
          test_loss = self.evaluate(train_data, valid_set)
          print('Epoch %s Train Loss %s Test Loss %s' % (epoch, train_loss, test_loss))
    
    def evaluate(self, train_data, valid_set):

        N, k = train_data.shape
        X_feature_test, y_feature_test = valid_set

        test_size = len(y_feature_test)
        test_loss = 0

        for i in range(test_size):
            b = copy.deepcopy(train_data)
            zero_ind = np.where(X_feature_test[i]==0)[0]
            b[zero_ind] = np.zeros(k)
            batch_X, batch_y = torch.FloatTensor(b).cuda(), torch.FloatTensor(y_feature_test[i:i+1]).cuda()
            batch_X, batch_y = batch_X.reshape((1, N, k)), batch_y.reshape((1, 1))
            y_pred = self.model(batch_X)
            loss = self.l2(y_pred, batch_y)
            loss_val = np.asscalar(loss.data.cpu().numpy())
            test_loss += loss_val
        test_loss /= test_size
        return test_loss
