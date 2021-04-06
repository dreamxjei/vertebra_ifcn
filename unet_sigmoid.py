# This code is provided for Deep Learning (CS 482/682) Homework 6 practice.
# The network structure is a simplified U-net. You need to finish the last layers
# @Copyright Cong Gao, the Johns Hopkins University, cgao11@jhu.edu
# Modified by Hongtao Wu on Oct 11, 2019 for Fall 2019 Machine Learning: Deep Learning HW6

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np


# Functions for adding the convolution layer
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    # Use batch normalization
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    # No batch normalization
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )


# Upsampling
def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )


# U-Net
class unet(nn.Module):
  def __init__(self, num_classes, useBN=False):
    super(unet, self).__init__()
    # Downgrade stages
    self.num_classes = num_classes
    self.conv1 = add_conv_stage(3, 32, useBN=useBN)
    self.conv2 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4 = add_conv_stage(128, 256, useBN=useBN)
    # Upgrade stages
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128,  64, useBN=useBN)
    self.conv1m = add_conv_stage( 64,  32, useBN=useBN)
    # Maxpool
    self.max_pool = nn.MaxPool2d(2)
    # Upsample layers
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128,  64)
    self.upsample21 = upsample(64 ,  32)
    # weight initialization
    # You can have your own weight intialization. This is just an example.
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
          m.bias.data.zero_()

    #TODO: Design your last layer & activations
    self.last_layer = nn.Sequential(
      nn.Conv2d(32, self.num_classes, 1),
#       nn.Softmax(dim=1) # softmax over classes
      nn.Linear(self.num_classes, 1)
    )
    
    self.last_layer_sigmoid = nn.Sequential(
      nn.Conv2d(32, self.num_classes, 1)
    )
    
    

  def forward(self, x):
#     print('x size is:', np.shape(x))  # torch.Size([5, 3, 640, 320]) | torch.Size([10, 3, 256, 320])
    conv1_out = self.conv1(x)
#     print('conv1_out size is:', np.shape(conv1_out))  # [5, 32, 640, 320] | [10, 32, 256, 320]
    conv2_out = self.conv2(self.max_pool(conv1_out))
#     print('conv2_out size is:', np.shape(conv2_out))  # [5, 64, 320, 160] | [10, 64, 128, 160]
    conv3_out = self.conv3(self.max_pool(conv2_out))
#     print('conv3_out size is:', np.shape(conv3_out))  # [5, 128, 160, 80] | [10, 128, 64, 80]
    conv4_out = self.conv4(self.max_pool(conv3_out))
#     print('conv4_out size is:', np.shape(conv4_out))  # [5, 256, 80, 40] | [10, 256, 32, 40]

    conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
#     print('conv4m_out_ size is:', np.shape(conv4m_out_))  # [5, 256, 160, 80] | [10, 256, 64, 80]
    conv3m_out  = self.conv3m(conv4m_out_)
#     print('conv3m_out size is:', np.shape(conv3m_out))  # [5, 128, 160, 80] | [10, 128, 64, 80]

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
#     print('conv3m_out_ size is:', np.shape(conv3m_out_))  # [5, 128, 320, 160] | [10, 128, 128, 160]
    conv2m_out  = self.conv2m(conv3m_out_)
#     print('conv2m_out size is:', np.shape(conv2m_out))  # [5, 64, 320, 160] | [10, 64, 128, 160]

    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
#     print('conv2m_out_ size is:', np.shape(conv2m_out_))  # [5, 64, 640, 320] | [10, 64, 256, 320]
    conv1m_out  = self.conv1m(conv2m_out_)
#     print('conv1m_out size is:', np.shape(conv1m_out))  # [5, 32, 640, 320] | [10, 32, 256, 320]

    #TODO: Design your last layer & activations
#     last_layer_output = self.last_layer(conv1m_out)
    last_layer_output = self.last_layer_sigmoid(conv1m_out)
    last_layer_output = torch.sigmoid(last_layer_output)

#     print('last_layer_output size is:',np.shape(last_layer_output))  # [5, 26, 640, 320] | [10, 8, 256, 320]

    # classification path! splits from conv4_out
    class_128 = add_conv_stage(self.maxpool(conv4_out))  # 128?
    class_64 = add_conv_stage(self.maxpool(class_128))  # lol wtf is this
    class_32 = add_conv_stage(self.maxpool(class_64))
    class_16 = add_conv_stage(self.maxpool(class_32))
    class_8 = add_conv_stage(self.maxpool(class_16))
    class_4 = add_conv_stage(self.maxpool(class_8))
    class_2 = add_conv_stage(self.maxpool(class_4))
    class_1 = self.maxpool(class_2)
    
    return last_layer_output
