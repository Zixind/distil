import pandas as pd 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('../')
from active_learning_strategies import FASS, EntropySampling, EntropySamplingDropout, RandomSampling,\
                                LeastConfidence,LeastConfidenceDropout, MarginSampling, MarginSamplingDropout, \
                                CoreSet, GLISTER, BADGE
from utils.models.logreg_net import LogisticRegNet
from utils.models.simpleNN_net import TwoLayerNet

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

#custom training
class data_train:

    def __init__(self, X, Y, net, handler, args):

        self.X = X
        self.Y = Y
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.use_cuda = torch.cuda.is_available()

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def update_data(self, X, Y):
    	self.X = X
    	self.Y = Y

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            if self.use_cuda:
                x, y = Variable(x.cuda()), Variable(y.cuda())
            else:
                x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            out = self.clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X)

    
    def train(self):

        print('Training..')
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        if self.use_cuda:
            self.clf =  self.net.apply(weight_reset).cuda()
        else:
            self.clf =  self.net.apply(weight_reset)

        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
        loader_tr = DataLoader(self.handler(self.X, self.Y, False))
        epoch = 1
        accCurrent = 0
        while accCurrent < 0.95 and epoch < n_epoch: 
            accCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            # print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        print('Epoch:', str(epoch),'Training accuracy:',round(accCurrent, 3), flush=True)
        return self.clf


class DataHandler_Points(Dataset):
    def __init__(self, X, Y=None, select=True):
        
        self.select = select
        if not self.select:
        	self.X = X.astype(np.float32)
        	self.Y = Y
        else:
        	self.X = X.astype(np.float32)  #For unlabeled Data

    def __getitem__(self, index):
    	if not self.select:
    		x, y = self.X[index], self.Y[index]
    		return x, y, index
    	else:
        	x = self.X[index]              #For unlabeled Data
        	return x, index

    def __len__(self):
        return len(self.X)

#User Execution
data_path = '../datasets/iris.csv'
test_path = '../datasets/iris_test.csv'
args = {'n_epoch':150, 'lr':float(0.001)}  #Different args than strategy_args
nclasses = 3    ##Number of unique classes
n_rounds = 11    ##Number of rounds to run active learning
budget = 10 		##Number of new data points after every iteration
strategy_args = {'batch_size' : 2, 'lr' : 0.1} 

df = pd.read_csv(data_path)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

X_tr = X[:20]    #First set of labeled examples
y_tr = y[:20]

X_unlabeled = X[20:]		#Unlabeled data
y_unlabeled = y[20:]			

df_test = pd.read_csv(test_path)
X_test = df_test.iloc[:,:-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

nSamps, dim = np.shape(X)
# net = mlpMod(dim, nclasses, embSize=3)
print('Dim',dim)
net = TwoLayerNet(dim, nclasses, dim*2)
net.apply(init_weights)

strategy_args = {'batch_size' : 2, 'submod' : 'facility_location', 'selection_type' : 'PerClass'} 
strategy = BADGE(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 2}
# strategy = EntropySampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)
# strategy = RandomSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = LeastConfidence(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = MarginSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)

# strategy_args = {'batch_size' : 2, 'n_drop' : 2}
# strategy = EntropySamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = LeastConfidenceDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = MarginSamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 1, 'tor':1e-4}
# strategy = CoreSet(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

#Training first set of points
dt = data_train(X_tr, y_tr, net, DataHandler_Points, args)
clf = dt.train()
strategy.update_model(clf)
y_pred = strategy.predict(X_test).numpy()

acc = np.zeros(n_rounds)
acc[0] = (1.0*(y_test == y_pred)).sum().item() / len(y_test)
print('Initial Testing accuracy:', round(acc[0], 3), flush=True)

##User Controlled Loop
for rd in range(1, n_rounds):
    print('-------------------------------------------------')
    print('Round', rd) 
    print('-------------------------------------------------')
    idx = strategy.select(budget)
    print('New data points added -', len(idx))
    strategy.save_state()

    #Adding new points to training set
    X_tr = np.concatenate((X_tr, X_unlabeled[idx]), axis=0)
    X_unlabeled = np.delete(X_unlabeled, idx, axis = 0)

    #Human In Loop, Assuming user adds new labels here
    y_tr = np.concatenate((y_tr, y_unlabeled[idx]), axis = 0)
    y_unlabeled = np.delete(y_unlabeled, idx, axis = 0)
    print('Number of training points -',X_tr.shape[0])
    print('Number of labels -', y_tr.shape[0])
    print('Number of unlabeled points -', X_unlabeled.shape[0])

    #Reload state and start training
    strategy.load_state()
    strategy.update_data(X_tr, y_tr, X_unlabeled)
    dt.update_data(X_tr, y_tr)

    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(X_test).numpy()
    acc[rd] = round(1.0 * (y_test == y_pred).sum().item() / len(y_test), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    # if acc[rd] > 0.98:
    #     print('Testing accuracy reached above 98%, stopping training!')
    #     break
print('Training Completed')
# final_df = pd.DataFrame(X_tr)
# final_df['Target'] = list(y_tr)
# final_df.to_csv('../final.csv', index=False)