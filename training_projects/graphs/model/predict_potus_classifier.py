import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import functools
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
import numpy as np
from init_func import *




class MultiInitialisationClassifier(BaseEstimator, ClassifierMixin):
    """ Sklearn-compatible classifier that implement a pytorch feedforward neural network
    and tries several weight initialisation in its "fit" method"""

    def __init__(self, net_class=nn.Sequential, criterion=nn.NLLLoss(), batch_size=32, n_epochs=10,
                 n_inits_per_optimizer=15, max_fail=4, dropout=(0, 0), init_function=init_weight_relu,
                 validation_set=None,
                 optimizer_classes={"Adadelta": functools.partial(optim.Adadelta, weight_decay=0.01)}):

        self.trained_net = None
        self.dropout = dropout
        self.p_init = dropout[0]
        self.p_hidden = dropout[1]
        self.criterion = criterion
        self.net_class = net_class
        np.unique
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_inits_per_optimizer = n_inits_per_optimizer
        self.max_fail = max_fail
        self.optimizer_classes = optimizer_classes
        self.init_function = init_function
        self.validation_set = validation_set

    def multi_init(self):
        net_list = list()
        for i, key in enumerate(self.optimizer_classes.keys()):
            for k in range(self.n_inits_per_optimizer):
                net = self.net_class_init()
                net.apply(self.init_function)
                # net.apply(init_uniform)
                net.train()
                net_list.append([net, self.optimizer_classes[key](net.parameters()), 0.0])
        return net_list

    def load_data(self, dataset):
        trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return trainloader

    def fit(self, X, y, n_epochs=None,
            validation_set=None, max_fail=5, thresold_fail=1.5):

        print(self.dropout)
        print(self.batch_size)
        print(self.optimizer_classes)
        self.p_init = self.dropout[0]
        self.p_hidden = self.dropout[1]
        self.net_class_init = functools.partial(self.net_class, n_features=X.shape[1], n_classes=len(np.unique(y)),
                                                dropout=self.dropout)
        if n_epochs == None:
            n_epochs = self.n_epochs
        if validation_set == None:
            validation_set = self.validation_set

        ## Load the data
        input, target = torch.from_numpy(X), torch.from_numpy(y)
        dataset = TensorDataset(input, target)
        trainloader = self.load_data(dataset)

        ## Create the different initialisation (i.e create different nets)
        net_list = self.multi_init()

        ## Train the nets
        print("BEGINNING TRAINING")
        loss_list = list()
        loss_list_validation = list()
        best_net_on_validation = [None, np.inf]
        for epoch in range(n_epochs):
            for net in net_list:
                net[2] = 0.0
                n_iter = 0
            for data in trainloader:
                input, target = data
                target = target.view(-1)
                input, target = Variable(input).float(), Variable(target).long()
                for i in range(len(net_list)):
                    net, optimizer, _ = net_list[i]
                    optimizer.zero_grad()
                    output = net(input)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    net_list[i][2] += loss.data[0]
                n_iter += 1
            index_min = np.nanargmin([net[2] for net in net_list])
            print(net_list[index_min][2] / n_iter)
            loss_list.append(net_list[index_min][2] / n_iter)
            ## Check the loss on the validation set and stop if it's increasing for at least max_fail epochs
            #  or go back above thresold_fail * (minimum loss attained on the validation set)
            if validation_set:
                # copy the best net and make it in evaluation mode for scoring
                best_net = copy.deepcopy(net_list[index_min][0])
                weight_scaling_inference_rule(best_net, self.p_init, self.p_hidden)
                best_net.eval()
                score = self.score(validation_set[0], validation_set[1], best_net)
                print(score)
                loss_list_validation.append(score)
                if score < best_net_on_validation[1]:  # WARNING : loss or score ???
                    best_net_on_validation = [best_net, score]
                if (score > thresold_fail * best_net_on_validation[1] or
                                max_fail < len(loss_list_validation)
                        and (np.array([loss_list_validation[i] - loss_list_validation[i - 1] for i in
                                       range(- 1, - max_fail - 1, - 1)]) > 0).all()
                    ):
                    print("EARLY STOPPING")
                    self.trained_net = best_net_on_validation[0]
                    return loss_list, loss_list_validation
            ##

            if len(net_list) > 1:
                del net_list[np.argmax([net[2] for net in net_list])]
            print(epoch)

        self.trained_net = best_net_on_validation[0]
        # self.trained_net = net_list[-1][0] # for training on train
        l = loss_list, loss_list_validation  # before
        # return self #for sklearn
        return l

    def predict(self, X):
        if not self.trained_net:
            raise RuntimeError("You must train the classifier before predicting !")
        previous_state = self.trained_net.training
        self.trained_net.eval()
        input = Variable(torch.from_numpy(X)).float()
        output = self.trained_net(input)
        predicted = torch.max(output.data, 1)[1]
        self.trained_net.training = previous_state
        return predicted.numpy()

    def predict_proba(self, X, net=None):
        if not net and not self.trained_net:
            raise RuntimeError("You must train the classifier before predicting !")
        if not net:
            net = self.trained_net
        previous_state = net.training
        net.eval()
        input = Variable(torch.from_numpy(X)).float()
        output = net(input)
        predicted = output.data
        net.training = previous_state
        return predicted.numpy()

    def score(self, X, y, net=None):
        if not net and not self.trained_net:
            raise RuntimeError("You must train the classifier before scoring !")
        if not net:
            net = self.trained_net
        previous_state = net.training
        net.eval()
        input, target = Variable(torch.from_numpy(X)).float(), Variable(torch.from_numpy(y)).long()
        output = net(input)
        net.training = previous_state
        return self.criterion(output, target).data[0]


