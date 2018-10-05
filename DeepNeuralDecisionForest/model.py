# https://github.com/jingxil/Neural-Decision-Forests/blob/master/ndf.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from functools import reduce


class Tree(nn.Module):
    def __init__(self,depth,n_in_feature,used_feature_rate,n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        self.feature_Layer = nn.Linear(1024, n_in_feature)
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
       
        self.feature_mask = self.feature_mask.cuda()
        x = x.cuda()

        feature_layer = self.feature_Layer(x)
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
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
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
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu,pi)
        return p


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




class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size()[0],-1)
        out = self.forest(out)
        return out

def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res

def torch_bin(x, cut_points, temp=0.1):
    D = cut_points.shape[0]
    W = torch.reshape(torch.linespace(1.0, D + 1.0, D + 1), [1,-1])
    cut_points , _ = torch.sort(cut_points)
    b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
    h = torch.matmul(x, W) + b
    res = torch.exp(h - torch.max(h))
    res = res/troch.sum(res, dim=-1, keepdim=True)
    return h

def nn_decision_tree(x, cut_points_list, leaf_score, temp=0.1):
    leaf = reduce(torch_kron_prod, map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], temp), enumerate(cut_points_list)))
    return torch.matmul(leaf, leaf_score)

class NNForest(nn.Module):
    def __init__(self, num_cut, num_leaf, num_class):
        super(NNForest, self).__init__()
        self.cut_points_list = [torch.rand([i], requires_grad=True) for i in num_cut]
        num_leaf = np.prod(np.array(num_cut) + 1)
        self.leaf_score = torch.rand([num_leaf, num_class], requires_grad=True)
        
    def forward(self, x):
        return nn_decision_tree(x, self.cut_points_list, self.leaf_score)


