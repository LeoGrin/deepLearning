import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import numpy as np
import pandas as pd

class RNN2(nn.Module):
    """
    Simple LSTM used to learn on text 
    TODO : use more complex architecture (bidirectional, more layers, ...)
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        #self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=0.1)
        self.lin = nn.Linear(hidden_size, output_size)
        #self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        #self.o2o = nn.Linear(hidden_size + output_size, output_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        #hidden = self.i2h(input_combined)
        #output = self.i2o(input_combined)
        #output_combined = torch.cat((hidden, output), 1)
        output, hidden = self.lstm(input.view(len(input), 1, -1), hidden)
        output = self.lin(output.view(1, -1))
        #output = self.softmax(output)
        return output, hidden


    def initHidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.n_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, 1, self.hidden_size)))


class USERNAME_ENCODER(nn.Module):
    """
    Use to save username as a vector representation. Should be used with the
    RNN_WITH_USERNAME class
    """
    def __init__(self, input_size, hidden_size , n_layers=1):
        super(USERNAME_ENCODER, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        #self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=0.1)
        #self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        #self.o2o = nn.Linear(hidden_size + output_size, output_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        #hidden = self.i2h(input_combined)
        #output = self.i2o(input_combined)
        #output_combined = torch.cat((hidden, output), 1)
        output, hidden = self.lstm(input.view(len(input), 1, -1), hidden)
        #output = self.softmax(output)
        return output, hidden


    def initHidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.n_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, 1, self.hidden_size)))


class RNN_WITH_USERNAME(nn.Module):
    """
    Simple LSTM to predict a text linked with an username with the username as an additional parameter
    (the username should be encoded as a vector by the USERNAME_ENCODER class and passed
    as a parameter to this RNN with each letter of the matching parameter)
    """
    def __init__(self, input_size, hidden_size_encoded_username, hidden_size, output_size, n_layers=1):
        super(RNN_WITH_USERNAME, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        #self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size_encoded_username + input_size, hidden_size, n_layers, dropout=0.1)
        self.lin = nn.Linear(hidden_size, output_size)
        #self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        #self.o2o = nn.Linear(hidden_size + output_size, output_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, username_tensor, input_tensor, hidden):
        #hidden = self.i2h(input_combined)
        #output = self.i2o(input_combined)
        input_combined = torch.cat((username_tensor, input_tensor), 1)
        output, hidden = self.lstm(input_combined.view(len(input_combined), 1, -1), hidden)
        output = self.lin(F.dropout(output.view(1, -1), 0.1, training=self.training))
        #output = self.softmax(output)
        return output, hidden


    def initHidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.n_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, 1, self.hidden_size)))

if __name__=="__main__":
    print(torch.cuda.is_available())