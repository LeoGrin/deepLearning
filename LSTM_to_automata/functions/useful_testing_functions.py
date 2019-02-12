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
from basic_functions import *

char_vect = np.array(['j', 'a', 'k', 'c', 'i', 'e', '\r', 'm', 'l', 'x', 'f', 'z', 'b', 'o', 'r', 'g', 'h', 'p',
 'n' ,'d' ,'1', '2' ,'8' ,'7' ,'3', 't', '4', 'y' ,'5' ,'0','9', 's', '6', 'u', 'P', 'G',
 'w' ,'v' ,'q', 'N', 'W' ,'T' ,'H', 'A', 'B', '.', 'L', 'X' ,'I' ,'#' ,'V' ,'C' ,'R', 'D',
 'F' ,'E' ,'Q' ,'M' ,'-' ,'_' ,'Y' ,'Z', 'K' ,'J' ,'O' ,'S' ,'U' ,'*', '%' ,'?' ,'!', '&',
 '$' ,'@' ,'=' ,';' ,' ', '\\', '`' ,'\x03', '|', '+', "'", '^', '[', '}', '~', '(', '\xaa',
 '\xbb' ,']', '/', '\xac' ,')', ':' ,'{' ,'\xe9' ,'\xbc', '\x0f' ,'>', '<' ,'\xa6',
 '\x02' ,'\x17', '\x07'])

def encode(username, encoder):
    encoder.eval()
    hidden_username = encoder.initHidden()
    username_tensor = Variable(text2input(username))
    for i in range(username_tensor.size()[0]):
        output, hidden_username = encoder(username_tensor[i], hidden_username)
    username_vect = hidden_username[0][-1]
    return username_vect

def sample(rnn, start, temp, encoder=None, username=None, username_vect=None, return_proba=False, coef_if_start_token=2, max_length=50):
    """ basic sample from model with modulable temperature"""
    if return_proba and temp != 1:
        print("Warning : temperature != 1 means returned proba not accurate")
    rnn.eval()
    hidden = rnn.initHidden()
    if encoder:
        encoder.eval()
    if username:
        hidden_username = encoder.initHidden()
        username_tensor = Variable(text2input(username))
        for i in range(username_tensor.size()[0]):
            output, hidden_username = encoder(username_tensor[i], hidden_username)

        username_vect = hidden_username[0][-1]
    start_input = Variable(text2input(start))
    if len(start_input) > 1:
        for i in range(len(start_input) - 1):
            output, hidden = rnn(username_vect, start_input[i], hidden)

    input = start_input[-1]
    sample_proba = 1

    output_text = start

    for i in range(max_length):
        if not username_vect:
            output, hidden = rnn(input, hidden)
        else:
            output, hidden = rnn(username_vect, input, hidden)
        output = output.data.view(-1)
        output_distrib = F.softmax(Variable((output - torch.ones(output.size()[0]) * torch.mean(output)) / temp))
        topi = torch.multinomial(output_distrib, 1)[0].data[0]
        sample_proba *= output_distrib[topi].data[0]

        # topi = output.data.topk(1)[1][0][0]
        if topi == len(char_vect):
            break
        elif topi == len(char_vect) + 1:
            print("ERROR : start token ! Sampling again from scratch with twice the temperature")
            return sample(rnn, start, temp * coef_if_start_token, encoder, username, username_vect, return_proba)
        else:
            char = char_vect[topi]
            output_text += char
            input = Variable(text2input(char)[1])  # Warning : remove start token
        if i == max_length - 1:
            print("STOPPED")
    if not return_proba:
        return output_text
    else:
        return output_text, sample_proba

# Monte Carlo estimator of nb_guess before guessing a password, assuming one can generate password in descending
# probability order (according to the model). The probabilty --> nb_guess relationship estimation is precomputed
# for computations gains.

# useful fonctions for the MC estimator

def sample_from_model(model, username_vect=None):
    return sample(model, "", 1, username_vect=username_vect, return_proba=True)

def precompute_sample_proba(model, n, username_vect=None):
    """precomputation for monte carlo acceleration"""
    # WARNING : the precomputed sample should match the samples later used for monte carlo estimation (e.g same username)
    A = np.zeros(n)
    for i in range(n):
        A[i] = sample_from_model(model, username_vect)[1]
    A = np.sort(A)[::-1]
    C = np.zeros(n)
    C[0] = 0
    for i in range(n - 1):
        C[i + 1] = C[i] + (1 / (n * A[i + 1]))
    return A, C

def compute_password_proba(model, password, username_vect=None):
    proba = 1
    hidden = model.initHidden()
    for char in password:
        char_index = np.where(char_vect == char)[0][0]
        if username_vect:
            output, hidden = model(username_vect, Variable(text2input(char)[-1]), hidden)
        else:
            output, hidden = model(Variable(text2input(char)[-1]), hidden)
        output = F.softmax(output)
        proba *= output[0][char_index].data[0]
    return proba

def nb_guess(model, password, precomputed_sample, username_vect=None):
    """Monte Carlo estimator of nb_guess before guessing a password"""
    password_proba = compute_password_proba(model, password, username_vect)
    A, C = precomputed_sample
    n = len(A)
    a = 0
    b = len(A) - 1
    i = 0
    while i < 10000:
        index = (a + b) / 2
        if password_proba < A[index]:
            a = index
        elif password_proba >= A[index]:
            b = index
        if abs(b - a) == 1:
            return C[b]
        i += 1

# Thanks to these fonctions, we are able to construct several metrics to compute the efficiency of our model.
def mean_nb_guess_with_username(rnn, password_list, encoder, n_iters=10):
    nb_guess_list = list()
    for i in range(n_iters):
        username, password = randomExample(password_list, True)
        username_vect = encode(username, encoder)
        precomputed_sample = precompute_sample_proba(rnn, 2000, username_vect)
        nb_guess_list.append(nb_guess(rnn, password, precomputed_sample, username_vect))
    print("mean log(nb_guess) : {}".format(np.mean(np.log(nb_guess_list))))
    print("median log(nb_guess) : {}".format(np.median(np.log(nb_guess_list))))
    print("standard deviation of log : {}".format(np.std(np.log(nb_guess_list))))
    return nb_guess_list
def mean_nb_guess(model, password_list, precomputed_sample, n_iters=1000):
    nb_guess_list = list()
    for i in range(n_iters):
        password = randomExample(password_list)
        nb_guess_list.append(nb_guess(model, password, precomputed_sample))
    print("mean log(nb_guess) : {}".format(np.mean(np.log(nb_guess_list))))
    print("median log(nb_guess) : {}".format(np.median(np.log(nb_guess_list))))
    print("standard deviation of log : {}".format(np.std(np.log(nb_guess_list))))
    return nb_guess_list

def recall_curve_monte_carlo(model_list, password_list, precomputed_sample_list, n_examples):
    color_list = ["blue", "red"]
    for k, model in enumerate(model_list):
        precomputed_sample = precomputed_sample_list[k]
        nb_guess_list = list()
        for i in range(n_examples):
            password = randomExample(password_list)
            nb_guess_list.append(nb_guess(model, password, precomputed_sample))
        nb_guess_list = sorted(nb_guess_list)
        plt.plot(np.log2(nb_guess_list), [100 * i / n_examples for i in range(n_examples)], c=color_list[k])
    plt.title("recall curve on {} examples".format(n_examples))
    plt.xlabel("log2(nb_guess)")
    plt.ylabel("percentage found")
    plt.show()