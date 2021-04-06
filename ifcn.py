# The network structure is based off of the IFCN paper, which itself is based off U-Net.
# @Copyright Jinchi Wei, the Johns Hopkins University, jwei9@jh.edu

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
    self.conv1 = add_conv_stage(29, 32, useBN=useBN)  # used to be 3, 32 - added instance mem
    self.conv2 = add_conv_stage(32, 64, useBN=useBN)
    self.conv3 = add_conv_stage(64, 128, useBN=useBN)
    self.conv4 = add_conv_stage(128, 256, useBN=useBN)
    # Upgrade stages
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128,  64, useBN=useBN)
    self.conv1m = add_conv_stage( 64,  32, useBN=useBN)
    # Maxpool
    self.max_pool = nn.MaxPool2d(2)  # kernel_size = 2 and stride automatically is 2
    # Upsample layers
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128,  64)
    self.upsample21 = upsample(64 ,  32)
    # dense (linear) layer for classification
    self.class_conv = add_conv_stage(32, 32, useBN=useBN)
    self.dense = nn.Linear(32, 1)  # github example uses num_channels, should be # in first layer
    # weight initialization
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
    

  def forward(self, x):  # all dimensions are potentials
    # print('x size is:', np.shape(x))  # torch.Size([10, 29, 128, 128])  (ins (26) concatenated with img (3))
    conv1_out = self.conv1(x)  # [10, 32, 128, 128]
    # print('conv1_out size is:', np.shape(conv1_out))  # [10, 32, 128, 128]
    conv2_out = self.conv2(self.max_pool(conv1_out))
    # print('conv2_out size is:', np.shape(conv2_out))  # [10, 64, 64, 64]
    conv3_out = self.conv3(self.max_pool(conv2_out))
    # print('conv3_out size is:', np.shape(conv3_out))  # [10, 128, 32, 32] | [10, 128, 64, 80]
    conv4_out = self.conv4(self.max_pool(conv3_out))
    # print('conv4_out size is:', np.shape(conv4_out))  # [10, 256, 16, 16] | [10, 256, 32, 40]

    conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
    # print('conv4m_out_ size is:', np.shape(conv4m_out_))  # [10, 256, 32, 32] | [10, 256, 64, 80]
    conv3m_out  = self.conv3m(conv4m_out_)
    # print('conv3m_out size is:', np.shape(conv3m_out))  # [10, 128, 32, 32] | [10, 128, 64, 80]

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    # print('conv3m_out_ size is:', np.shape(conv3m_out_))  # [10, 128, 64, 64] | [10, 128, 128, 160]
    conv2m_out  = self.conv2m(conv3m_out_)
    # print('conv2m_out size is:', np.shape(conv2m_out))  # [10, 64, 64, 64] | [10, 64, 128, 160]

    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
    # print('conv2m_out_ size is:', np.shape(conv2m_out_))  # [10, 64, 128, 128 | [10, 64, 256, 320]
    conv1m_out  = self.conv1m(conv2m_out_)
    # print('conv1m_out size is:', np.shape(conv1m_out))  # [10, 32, 128, 128] | [10, 32, 256, 320]

    # Design last layer & activations
    last_layer_output = self.last_layer_sigmoid(conv1m_out)  # should be [10, 26, 128, 128]
    last_layer_output = torch.sigmoid(last_layer_output)

    # print('last_layer_output size is:',np.shape(last_layer_output))  # [10, 26, 128, 128] | [10, 8, 256, 320]

    # classification path! splits from conv4_out
    class_8 = self.conv3m(self.max_pool(conv4_out))
    # print('class_8 size is:',np.shape(class_8))  # [10, 128, 8, 8]
    class_4 = self.conv2m(self.max_pool(class_8))
    # print('class_4 size is:',np.shape(class_4))  # [10, 64, 4, 4]
    class_2 = self.conv1m(self.max_pool(class_4))
    # print('class_2 size is:',np.shape(class_2))  # [10, 32, 2, 2]
    class_1 = self.max_pool(class_2)
    # print('class_1 size is:',np.shape(class_1))  # [10, 32, 1, 1]

    # print('torch.flatten(class_1) size is:',np.shape(torch.flatten(class_1, start_dim=1)))  # [10, 32]
    class_output = torch.sigmoid(self.dense(torch.flatten(class_1, start_dim=1)))  # not sure what size this should be
    # class_output = torch.sigmoid(self.dense(class_1))
    # print('class_output size is:',np.shape(class_output))  # [10, 1]
    
    return last_layer_output, class_output
