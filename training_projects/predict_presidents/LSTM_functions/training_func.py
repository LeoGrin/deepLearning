import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(input_line_tensor, target_line_tensor, rnn, criterion, optimizer):
    rnn.train()
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()
    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]


def training_iters(n_iters, rnn, examplesFunction, criterion=nn.CrossEntropyLoss(), optimizer_name="SGD",
                   print_every=100, plot_every=50,
                   lr=0.001, momentum=0.0001):
    try:
        optimizer_dict = {"Adadelta": optim.Adadelta(rnn.parameters()),
                          "SGD": optim.SGD(rnn.parameters(), lr, momentum), "Adam": optim.Adam(rnn.parameters(), lr)}
        optimizer = optimizer_dict[optimizer_name]
        n_iters = n_iters
        all_losses = []
        total_loss = 0  # Reset every plot_every iters
        rnn.train()
        start = time.time()

        for iter in range(1, n_iters + 1):
            output, loss = train(*examplesFunction(), rnn=rnn, criterion=criterion, optimizer=optimizer)
            total_loss += loss

            if iter % print_every == 0:
                print(
                    '%s (%d %d%%) %.4f' % (
                    timeSince(start), iter, float(iter) / n_iters * 100, total_loss / plot_every))

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0

        return all_losses
    except KeyboardInterrupt:
        plt.figure()
        plt.plot(all_losses)
        plt.show()
        return all_losses
