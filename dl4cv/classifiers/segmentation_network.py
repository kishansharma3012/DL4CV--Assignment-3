import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

class SegmentationNetwork(nn.Module):

    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        ############################################################################
        #                             YOUR CODE
        #                      #
        ############################################################################
        model_conv = models.resnet18(pretrained = True)
        for param in model_conv.parameters():
            param.requires_grad = False
        
        self.Res_conv = nn.Sequential(
                        model_conv.conv1,
                        model_conv.bn1,
                        model_conv.relu,
                        model_conv.maxpool,
                        model_conv.layer1,
                        model_conv.layer2,
                        model_conv.layer3,
                        model_conv.layer4,
                        )


        self.my_model = nn.Sequential(
                        nn.Conv2d(512,256,kernel_size=3, stride=1, padding=0),
                        nn.ReLU(),
                        nn.ConvTranspose2d(256,128,kernel_size=(5,5), stride=5, padding=(1,1), output_padding=(0,0)),
                        nn.BatchNorm2d(128,eps = 1e-05,momentum = 0.1, affine =True),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128,64,kernel_size=(3,3), stride=3, padding=(2,2), output_padding=(1,1)),
                        nn.BatchNorm2d(64,eps = 1e-05,momentum = 0.1, affine =True),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64,24,kernel_size=(3,3), stride=3, padding=(2,2), output_padding=(1,1))
                        )    
        for param in self.my_model.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ############################################################################
        #                             YOUR CODE                                    #
        ############################################################################
        out = self.Res_conv(x)
        out = self.my_model(out)
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
