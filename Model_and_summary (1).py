from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary

class Net(nn.Module):
  
  def __init__(self):
    super(Net, self).__init__()
    
    self.convblock1= nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), # RF 3
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=16, out_channels=32 , kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 5
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 7
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), # RF 9
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), #RF 11
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout2d(0.1)
    )
    
    self.transblock1= nn.Sequential(
        nn.MaxPool2d(2,2), #RF 12
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)
    )
    
    self.convblock2= nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 16
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 20
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), bias=False, padding=1, groups=128, padding_mode='same'), #RF 24
        nn.BatchNorm2d(256),
        nn.ReLU(), 
        nn.Dropout2d(0.1)
    )
    
    self.transblock2= nn.Sequential(
        nn.MaxPool2d(2,2), #26
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)
    )    
    self.convblock3= nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 34
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same'), #RF 42
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), bias=False, padding=1, padding_mode='same',dilation=2), #RF 42
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), bias=False, padding=1,groups=128, padding_mode='same'), #RF 50
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout2d(0.1)
    )
    
    self.transblock3= nn.Sequential(
        nn.MaxPool2d(2,2), #RF 54
        nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1,1), bias=False)
    )

    self.convblock4=nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'), #66
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(3),  # RF 74
        nn.Conv2d(in_channels=64,out_channels=10, kernel_size=(1,1),bias=False)
    )

    
    
  def forward(self, x):
      x=self.convblock1(x)
      x=self.transblock1(x)
      x=self.convblock2(x)
      x=self.transblock2(x)
      x=self.convblock3(x)
      x=self.transblock3(x)
      x=self.convblock4(x)
      x=x.view(-1,10)
      return F.log_softmax(x)
	
def disp_summary(model):
	summary(model, input_size=(3,32,32))