import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab, data, datasets
from torch.autograd import Variable
import main
import sys
import os
import numpy as np
import onnx
from model import Forest

parser = main.parser
GeneratorDevice = main.GeneratorDevice
DiscriminatorDevice = main.DiscriminatorDevice

dtype = torch.float32
args, unknown = parser.parse_known_args()
DataFolder = args.input
batchSize = args.batchsize
model = args.model
Scale = args.Scale
DeviceConfig = args.deviceConfig
gan = args.gan
tileData = args.tileData
validateData = args.validateData
testData = args.testData

dtype = torch.float32
glove = vocab.GloVe(name='840B', dim=300)

def get_glove_vec(word):
    return glove.vectors[glove.stoi[word]]

def get_word_from_vec(vec, n=10):
    all_dists = [(w,torch.dist(vec, get_glove_vec(w))) for w in glove.itos]
    return sorted(all_dist, key=lambda t: t[1])[:n]


################ Definition ######################### 
DEPTH = 10  # Depth of a tree
N_LABEL = 2  # Number of classe s
N_TREE = 50 # Number of trees (ensemble)
# network hyperparameters
p_conv_keep = 0.8
p_full_keep = 0.5
N_LEAF = 2 ** (DEPTH + 1)  # Number of leaf node

#896 430 

model = Forest(n_tree=N_TREE, tree_depth=DEPTH, n_in_feature=N_LEAF, tree_feature_rate=p_conv_keep, n_class=N_LABEL, jointly_training=True)

model = model.cuda()
#model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=.001) 

# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL, root='./.vector_cache')

# build the vocabulary
TEXT.build_vocab(train, vectors=glove)
LABEL.build_vocab(train)

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batchSize, device=0)
train_loader = train_iter
test_loader = test_iter

input = torch.zeros(batchSize, 800)

def train(epochs):

    for epoch in range(epochs):
        for i, dataTensor in enumerate(train_loader):  
           
            data = dataTensor.text[0]
            target = dataTensor.label.data
            target = torch.tensor(target, dtype=torch.long, device=GeneratorDevice)
            input.new_tensor(data, device=torch.device('cuda:0'))
            if(input.shape[0] == batchSize):

                optimizer.zero_grad()
                output = model(input)
                loss = F.nll_loss((output), target)
                loss.backward()
                optimizer.step()

                if i % 200 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i, len(train_loader)*batchSize, 100. * i / (len(train_loader)*batchSize), loss.item()))
                    
def test():
    test_loss = 0
    correct = 0
    for i, dataTensor in enumerate(test_loader):
        data = dataTensor.text[0]
        target = dataTensor.label.data

        target = torch.tensor(target, dtype=torch.long, device=GeneratorDevice)
        input.new_tensor(data, device=GeneratorDevice)
        if(input.shape[0] == batchSize):
            output = model(input)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

