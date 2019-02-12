import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import pandas as pd

char_vect = np.array(['j', 'a', 'k', 'c', 'i', 'e', '\r', 'm', 'l', 'x', 'f', 'z', 'b', 'o', 'r', 'g', 'h', 'p',
 'n' ,'d' ,'1', '2' ,'8' ,'7' ,'3', 't', '4', 'y' ,'5' ,'0','9', 's', '6', 'u', 'P', 'G',
 'w' ,'v' ,'q', 'N', 'W' ,'T' ,'H', 'A', 'B', '.', 'L', 'X' ,'I' ,'#' ,'V' ,'C' ,'R', 'D',
 'F' ,'E' ,'Q' ,'M' ,'-' ,'_' ,'Y' ,'Z', 'K' ,'J' ,'O' ,'S' ,'U' ,'*', '%' ,'?' ,'!', '&',
 '$' ,'@' ,'=' ,';' ,' ', '\\', '`' ,'\x03', '|', '+', "'", '^', '[', '}', '~', '(', '\xaa',
 '\xbb' ,']', '/', '\xac' ,')', ':' ,'{' ,'\xe9' ,'\xbc', '\x0f' ,'>', '<' ,'\xa6',
 '\x02' ,'\x17', '\x07'])

def randomExample(lines, with_username=False):
    n = len(lines)
    index = np.random.randint(0, n - 1)
    try:
        res = lines[index].split("\t")
        if not with_username:
            return res[1][:-1] #remove the "\r" at the end to put our own EOS instead
        else:
            return res[0], res[1][:-1]
    except:
        print(lines[index])
        return randomExample(lines)

def build_char_list(text_list):
    list_char = list()
    for text in text_list:
        text = text.split("\t")[-1]
        for char in text:
            if char not in list_char:
                list_char.append(char)
    char_vect = np.array(list_char)
    return char_vect


def trainingExample(text_list, with_username=False):
    if with_username:
        username, password = randomExample(text_list, with_username)
    else:
        password = randomExample(text_list)
    if len(password) > 0:
        train = text2input(password)
        target = text2target(password)
        if with_username:
            user_tensor = text2input(username)
            return Variable(
                user_tensor - float(np.mean(user_tensor.numpy())) / float(np.std(user_tensor.numpy()))), Variable(
                train - float(np.mean(train.numpy())) / float(np.std(train.numpy()))), Variable(target)
        else:
            return Variable(train - float(np.mean(train.numpy())) / float(np.std(train.numpy()))), Variable(target)
    else:
        return trainingExample(text_list, with_username)

    print("yep, definitely")
    return trainingExample(text_list, with_username)


def text2input(text):
    train_vec = torch.zeros(len(text) + 1, 1, len(char_vect) + 2)
    for i, char in enumerate(text):
        train_vec[i + 1][0][np.where(char_vect == char)[0][0]] = 1
    train_vec[0][0][len(char_vect) + 1] = 1

    return train_vec


def text2target(text):
    target_list = [np.where(char_vect == char)[0][0] for char in text]
    target_list.append(len(char_vect))
    return torch.LongTensor(target_list)

def clip_grad_norm(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm