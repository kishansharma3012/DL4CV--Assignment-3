from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def CrossEntropy2d(self,input, target, weight=None, pixel_average=True):
        # input:(n, c, h, w) target:(n, h, w)
        
        n, c, h, w = input.size()

        input = input.transpose(1, 2).transpose(2, 3).contiguous()
        input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

        target_mask = target >= 0
        target = target[target_mask]

        loss = F.cross_entropy(input, target, weight=weight, size_average=False)
        if pixel_average:
            loss /= target_mask.sum().data[0]
        return loss

    def train(self, model, train_loader, val_loader,log_nth=0,num_epochs=10):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.my_model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################
        for epoch in range(num_epochs):
            for i, data in enumerate(train_loader, 0):
                # get the inputs, wrap them in Variable
                input, label = data
                inputs, labels = Variable(input), Variable(label)
                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                
                loss = self.CrossEntropy2d(outputs, labels)
                
                loss.backward()
                optim.step()

                # Storing values
                self.train_loss_history.append(loss.data[0])
                if (i+1) % log_nth == 0:
                    print ('[Iteration %d/%d] Train loss: %0.4f') % \
                          (i, iter_per_epoch, self.train_loss_history[-1])

                if (i+1) % iter_per_epoch == 0:
                    _,predicted_train = torch.max(outputs.data, 1)
                    total_train = labels.size(0)
                    correct_train = (predicted_train == label).sum()
                    self.train_acc_history.append(correct_train/float(total_train))
                    print ('[Epoch %d/%d] Train acc/loss: %0.4f/%0.4f') % \
                          (epoch, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1])

            correct_val = 0
            val_size = 0
            loss_val = 0
            for i, data_val in enumerate(val_loader, 0):
                # get the inputs, wrap them in Variable
                input_val, label_val = data_val
                inputs_val, labels_val = Variable(input_val), Variable(label_val)
                output_val = model(inputs_val)
                loss_val = self.loss_func(output_val, labels_val)
                _,predicted_val = torch.max(output_val.data, 1)
                val_size += label_val.size(0)
                correct_val += (predicted_val == label_val).sum()
            self.val_acc_history.append(correct_val/float(val_size))
            print ('[Epoch %d/%d] Val acc/loss: %0.4f/%0.4f') % \
                  (epoch, num_epochs, self.val_acc_history[-1], loss_val.data[0])


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'

