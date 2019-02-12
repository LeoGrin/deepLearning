import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math

def L1_reg(model):
    l1_sum = None
    for W in model.parameters():
        if l1_sum is None:
            l1_sum = W.norm(1)
        else:
            l1_sum = l1_sum + W.norm(1) # WARNING : inplace operations on leaf variable not allowed
    return l1_sum



class L1CrossEntropyLoss():
    def __init__(self, rnn, L1_reg_param,  *args):
        self.criterion = nn.CrossEntropyLoss(*args)
        self.rnn = rnn
        self.L1_reg_param = L1_reg_param
    def __call__(self, output, target):
        return self.criterion(output, target) + L1_reg(self.rnn) * self.L1_reg_param

if __name__ == "__main__":
    pass
