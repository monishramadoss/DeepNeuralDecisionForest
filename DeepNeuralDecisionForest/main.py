from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import argparse
import torch

parser = argparse.ArgumentParser()
GeneratorDevice = torch.device('cpu')
DiscriminatorDevice = torch.device('cpu')

parser.add_argument('-i', "--input", type=str, 
                    default='./Images', 
                    help="Input directory where where training dataset and meta data are saved", 
                    required=False
                    )
parser.add_argument('-v',"--validateData", type=str,
                    default=None,
                    help="Validation Dir",
                    required=False
                    )
parser.add_argument('-t',"--testData", type=str,
                    default=None,
                    help="Testing Dir",
                    required=False
                    )                    
parser.add_argument('-m', "--model", type=int,
                    default=2,
                    help="SRDenseNet:0, SRResNet:1, SRAutoEncoder:2",
                    required=False
                    )
parser.add_argument('-g', "--gan", type=int,
                    default = 0,
                    help="Enable or Disable gan",
                    required=False
                    )
parser.add_argument('-e',"--epochs", type=int,
                    default=100,
                    help="Number of Epochs",
                    required=False
                    )
parser.add_argument('-bs',"--batchsize", type=int,
                    default=64,
                    help="Size of Mini-Batch",
                    required=False
                    )
parser.add_argument('-s',"--Scale", type=int,
                    default=2,
                    help="Scaling Factor",
                    required=False
                    )
parser.add_argument('-dc',"--deviceConfig", type=int,
                    default=1,
                    help="Device Configuration -> 0:(G:CPU D:CPU) 1:(G:GPU D:CPU) 2:(G:CPU D:GPU) 3:(G:GPU D:GPU)",
                    required=False
                    )
parser.add_argument('-td',"--tileData", type=int,
                    default=1,
                    help="Enable image tiling",
                    required=False
                    )


args, unknown = parser.parse_known_args()
model = args.model 
DeviceConfig = args.deviceConfig
Scale = args.Scale

if(DeviceConfig == 1):
    GeneratorDevice = torch.device('cuda:0')
if(DeviceConfig == 2):
    DiscriminatorDevice = torch.device('cuda:0')
if(DeviceConfig == 3):
    DiscriminatorDevice = torch.device('cuda:1')
    GeneratorDevice = torch.device('cuda:0')

import trainer



if __name__ == "__main__":
    trainer.train(10)
    trainer.test()
    exit(0)