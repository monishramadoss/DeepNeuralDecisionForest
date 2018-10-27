﻿
# https://github.com/jingxil/Neural-Decision-Forests/blob/master/ndf.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from functools import reduce


class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()

        self.conv3 = nn.Conv2d(1, 3, (3, 300))
        self.conv4 = nn.Conv2d(1, 4, (4, 300))
        self.conv5 = nn.Conv2d(1, 5, (5, 300))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in [self.conv3, self.conv4, self.conv5]]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)

        return x



class Tree(nn.Module):
    def __init__(self,depth,n_in_feature,used_feature_rate,n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        self.feature_Layer = nn.Linear(800, n_in_feature)
        # used features in this tree
        n_used_feature = int(n_in_feature*used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),requires_grad=False)
        # leaf label distribution
        if jointly_training:
            self.pi = np.random.rand(self.n_leaf,n_class)
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor),requires_grad=True)
        else:
            self.pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        # decision
        self.decision = nn.Sequential(OrderedDict([
                        ('linear1',nn.Linear(n_used_feature,self.n_leaf)),
                        ('sigmoid', nn.Sigmoid()),
                        ]))

    def forward(self,x):
        x = x.cuda()
        self.feature_mask = self.feature_mask
        feature_layer = self.feature_Layer(x).cuda()
        feats = torch.mm(feature_layer, self.feature_mask) # ->[batch_size,n_used_feature]
        decision = self.decision(feats) # ->[batch_size,n_leaf]
        decision = torch.unsqueeze(decision,dim=2)
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size,1,1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size,-1,1).repeat(1,1,2)
            # _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2] #16,1,2
            indices = torch.tensor([x for x in range(begin_idx, end_idx)]).cuda()
            _decision = torch.index_select(decision, 1, indices)
            _mu = _mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)

        mu = _mu.view(batch_size,self.n_leaf)

        return mu

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi,dim=-1)
        else:
            return self.pi

    def cal_prob(self,mu,pi):
        return torch.mm(mu, pi)


    def update_pi(self,new_pi):
        self.pi.data=new_pi


class Forest(nn.Module):
    def __init__(self,n_tree,tree_depth,n_in_feature,tree_feature_rate,n_class,jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree  = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth,n_in_feature,tree_feature_rate,n_class,jointly_training)
            self.trees.append(tree)

    def forward(self,x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p=tree.cal_prob(mu,tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs,dim=2)
        prob = torch.sum(probs,dim=2)/self.n_tree

        return prob



