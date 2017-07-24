import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride=1, weight_scale=0.001, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Stride for the convolution layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim

        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "same" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################
        self.first_layer = nn.Sequential(
            nn.Conv2d(channels, num_filters,kernel_size, stride, (kernel_size-1)/2), nn.ReLU(), nn.MaxPool2d(pool))

        input_size = num_filters*(height/pool)*(width/pool)

        self.second_layer = nn.Sequential(nn.Linear(input_size, hidden_dim), nn.Dropout(dropout), nn.ReLU() )
        self.third_layer = nn.Linear(hidden_dim, num_classes)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################
        out = self.first_layer(x)
        out = out.view(out.size(0), -1)
        out = self.second_layer(out)
        out = self.third_layer(out)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)
