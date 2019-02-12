import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
from basic_functions import *

char_vect = np.array(['j', 'a', 'k', 'c', 'i', 'e', '\r', 'm', 'l', 'x', 'f', 'z', 'b', 'o', 'r', 'g', 'h', 'p',
 'n' ,'d' ,'1', '2' ,'8' ,'7' ,'3', 't', '4', 'y' ,'5' ,'0','9', 's', '6', 'u', 'P', 'G',
 'w' ,'v' ,'q', 'N', 'W' ,'T' ,'H', 'A', 'B', '.', 'L', 'X' ,'I' ,'#' ,'V' ,'C' ,'R', 'D',
 'F' ,'E' ,'Q' ,'M' ,'-' ,'_' ,'Y' ,'Z', 'K' ,'J' ,'O' ,'S' ,'U' ,'*', '%' ,'?' ,'!', '&',
 '$' ,'@' ,'=' ,';' ,' ', '\\', '`' ,'\x03', '|', '+', "'", '^', '[', '}', '~', '(', '\xaa',
 '\xbb' ,']', '/', '\xac' ,')', ':' ,'{' ,'\xe9' ,'\xbc', '\x0f' ,'>', '<' ,'\xa6',
 '\x02' ,'\x17', '\x07'])

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

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)