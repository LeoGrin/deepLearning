import numpy as np

import torch.nn as nn




def init_weights(m, i):
    if type(m) == nn.Linear:
        m.weight.data.fill_(i)


def init_weight_normal(neuron, mean, std):
    if type(neuron) == nn.Linear:
        neuron.weight.data.normal_(mean, std)


def init_weight_relu(neuron):
    if type(neuron) == nn.Linear:
        n = len(neuron.weight.data)
        neuron.weight.data.normal_(0, np.sqrt(2. / n))


def init_normalized_bengio(neuron):
    if type(neuron) == nn.Linear:
        n_output, n_input = neuron.weight.data.numpy().shape
        neuron.weight.data.uniform_(0, np.sqrt(6 / (n_input + n_output)))


def init_uniform(neuron):
    if type(neuron) == nn.Linear:
        n_output, n_input = neuron.weight.data.numpy().shape
        neuron.weight.data.uniform_(0, np.sqrt(1 / (n_input)))


def normalize(vect):
    vect = (vect - np.mean(vect)) / np.std(vect)
    return vect


def weight_scaling_inference_rule(net, p_init, p_hidden):
    """Multiply the weight of each layer by its probability of not being droppped out during training to average bagging"""
    list(net.children())[0].weight.data *= (1 - p_init)
    for neuron in list(net.children())[1:]:
        if neuron.type == nn.Linear:
            neuron.weight.data *= (1 - p_hidden)
