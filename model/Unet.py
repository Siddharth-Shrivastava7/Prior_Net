import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
from torch.nn import Parameter as P
from torch.nn import init
import torch.optim as optim
import functools
import numpy as np


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels)
    )   

# def conv_deconv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=False),
#         nn.BatchNorm2d(out_channels),
#         nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=False),
#         nn.BatchNorm2d(out_channels)
#     )   

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64) #chg for 19 channel input tensor .. an exp 
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        # self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        # print(out.shape)
        
        # return torch.sigmoid(out) 
        return out

class UNet_mod(nn.Module):

    def __init__(self, n_class=2):
        super().__init__()
                
        self.dconv_down1 = double_conv(19, 64) #chg for 19 channel input tensor .. an exp 
        self.dconv_down2 = double_conv(64, 128)         
        self.dconv_down3 = double_conv(128, 256) 
        self.dconv_down4 = double_conv(256, 512) 
        self.dconv_down5 = double_conv(512, 512) 
        self.dconv_down6 = double_conv(512, 1024)       

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   # bilinear upsampling
         
        self.dconv_up6 = double_conv(1024, 512)
        self.upsample1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1) 
        self.dconv_up5 = double_conv(512*2, 512) 
        self.upsample2 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1) 
        self.dconv_up4 = double_conv(512*2, 256)
        self.upsample3 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.dconv_up3 = double_conv(256*2, 128)
        self.upsample4 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1) 
        self.dconv_up2 = double_conv(128*2, 64)
        self.upsample5 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.dconv_up1 = double_conv(64*2, 64)
        
        # self.conv_last = nn.Conv2d(64, n_class, 1)
        self.conv_last = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        conv1 = self.dconv_down1(x) #64
        x = self.maxpool(conv1) 

        conv2 = self.dconv_down2(x) #128
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x) #256
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x) #512 
        x = self.maxpool(conv4) 
        
        conv5 = self.dconv_down5(x) #512
        # print(conv5.shape)
        x = self.maxpool(conv5) 

        x = self.dconv_down6(x) #1024

        conv7 = self.dconv_up6(x)  #512   
        x = self.upsample(conv7) 
        # print(x.shape)
        x = torch.cat([x, conv5], dim=1)
        
        conv8 = self.dconv_up5(x)  #512     
        x = self.upsample(conv8) 
        x = torch.cat([x, conv4], dim=1)

        conv9 = self.dconv_up4(x)  #256      
        x = self.upsample(conv9)
        x = torch.cat([x, conv3], dim=1)

        conv10 = self.dconv_up3(x) #128     
        x = self.upsample(conv10)
        x = torch.cat([x, conv2], dim=1)

        conv11 = self.dconv_up2(x) #64  
        x = self.upsample(conv11)
        # print(x.shape)
        x = torch.cat([x, conv1], dim=1)
        # print(x.shape)

        x = self.dconv_up1(x)       
  
        out = self.conv_last(x)
        # print(out.shape)
        
        return torch.sigmoid(out) 
        # return out

class UNet_mod_2(nn.Module):

    def __init__(self, n_class=2):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64) #chg to 3 channel for an image input...comparing the tensor proj with the image
        self.dconv_down2 = double_conv(64, 128)         
        self.dconv_down3 = double_conv(128, 256) 
        self.dconv_down4 = double_conv(256, 512) 
        self.dconv_down5 = double_conv(512, 512) 
        self.dconv_down6 = double_conv(512, 1024)       

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   # bilinear upsampling
         
        self.dconv_up6 = double_conv(1024, 512)
        self.upsample1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1) 
        self.dconv_up5 = double_conv(512*2, 512) 
        self.upsample2 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1) 
        self.dconv_up4 = double_conv(512*2, 256)
        self.upsample3 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.dconv_up3 = double_conv(256*2, 128)
        self.upsample4 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1) 
        self.dconv_up2 = double_conv(128*2, 64)
        self.upsample5 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.dconv_up1 = double_conv(64*2, 64)
        
        # self.conv_last = nn.Conv2d(64, n_class, 1) # ce loss 
        self.conv_last = nn.Conv2d(64, 1, 1) # focal or bce loss 
        
    def forward(self, x):
        conv1 = self.dconv_down1(x) #64
        x = self.maxpool(conv1) 

        conv2 = self.dconv_down2(x) #128
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x) #256
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x) #512 
        x = self.maxpool(conv4) 
        
        conv5 = self.dconv_down5(x) #512
        # print(conv5.shape)
        x = self.maxpool(conv5) 

        x = self.dconv_down6(x) #1024

        conv7 = self.dconv_up6(x)  #512   
        x = self.upsample(conv7) 
        # print(x.shape)
        x = torch.cat([x, conv5], dim=1)
        
        conv8 = self.dconv_up5(x)  #512     
        x = self.upsample(conv8) 
        x = torch.cat([x, conv4], dim=1)

        conv9 = self.dconv_up4(x)  #256      
        x = self.upsample(conv9)
        x = torch.cat([x, conv3], dim=1)

        conv10 = self.dconv_up3(x) #128     
        x = self.upsample(conv10)
        x = torch.cat([x, conv2], dim=1)

        conv11 = self.dconv_up2(x) #64  
        x = self.upsample(conv11)
        # print(x.shape)
        x = torch.cat([x, conv1], dim=1)
        # print(x.shape)

        x = self.dconv_up1(x)       
  
        out = self.conv_last(x)
        # print(out.shape)
        
        return torch.sigmoid(out) 
        # return out

def Unet_model(num_classes = 2):
    model = UNet(num_classes)
    return model

def D_unet_arch(ch=64, attention='64',ksize='333333', dilation='111111',out_channel_multiplier=1):
    arch = {}
    n = 2
    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [3] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}


    arch[256] = {'in_channels' :            [19] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1         ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}
    # print('&&&&&&&&')
    # print(arch[256]['attention']) # Adding attention layer in D at resolution 64
    return arch 


class Conv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
  def forward(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    nn.Linear.__init__(self, in_features, out_features, bias)
  def forward(self, x):
    return F.linear(x, self.weight, self.bias) 


class Attention(nn.Module):
  def __init__(self, ch, which_conv=Conv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=Conv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x

  def forward(self, x):
    if self.preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)

    return h + self.shortcut(x)


class GBlock2(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, activation=None,
               upsample=None, skip_connection = True):
    super(GBlock2, self).__init__()

    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv = which_conv
    self.activation = activation
    self.upsample = upsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)

    # upsample layers
    self.upsample = upsample
    self.skip_connection = skip_connection

  def forward(self, x):
    h = self.activation(x)
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    #print(h.size())
    h = self.activation(h)
    h = self.conv2(h)
    # may be changed to h = self.conv2.forward_wo_sn(h)
    if self.learnable_sc:
      x = self.conv_sc(x)

    if self.skip_connection:
        out = h + x
    else:
        out = h
    return out

class Unet_Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                             D_kernel_size=3, D_attn='64', n_classes=1000,
                             D_activation=nn.ReLU(inplace=False),
                             D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                             output_dim=1, D_mixed_precision=False,
                             D_init='ortho', skip_init=False, decoder_skip_connection = True, **kwargs):
        super(Unet_Discriminator, self).__init__()

        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        # self.init = D_init

        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier  )[resolution]

        self.which_conv = functools.partial(Conv2d, kernel_size=3, padding=1)
        self.which_linear = functools.partial(Linear)

        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []    

        for index in range(len(self.arch['out_channels'])):
          # print(index)
          # print('*****') 
          if self.arch["downsample"][index]:
              self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                            out_channels=self.arch['out_channels'][index],
                                            which_conv=self.which_conv,
                                            wide=self.D_wide,
                                            activation=self.activation,
                                            preactivation=(index > 0),
                                            downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

          elif self.arch["upsample"][index]:
              upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                  if self.arch['upsample'][index] else None)

              self.blocks += [[GBlock2(in_channels=self.arch['in_channels'][index],
                                                        out_channels=self.arch['out_channels'][index],
                                                        which_conv=self.which_conv,
                                                        #which_bn=self.which_bn,
                                                        activation=self.activation,
                                                        upsample= upsample_function, skip_connection = True )]]

          # If attention on this block, attach it to the end
          # attention_condition = index < 5
          attention_condition = False
          if self.arch['attention'][self.arch['resolution'][index]] and attention_condition: #index < 5
              print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
              print("index = ", index)
              self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                                                        self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # print('******')
        # print(self.ch*self.out_channel_multiplier) #64
        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier, 2 ,kernel_size=1) # 2 channel outputs for using CE loss
        # last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier, 1 ,kernel_size=1) # 1 channel for focal loss
        self.blocks.append(last_layer)
  
        # Initialize weights
        # if not skip_init:
        #     self.init_weights()

        ###
        # print("_____params______")
        # for name, param in self.named_parameters():
        #     print(name, param.size())

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    def init_weights(self):
      self.param_count = 0
      for module in self.modules():
          if (isinstance(module, nn.Conv2d)
                  or isinstance(module, nn.Linear)
                  or isinstance(module, nn.Embedding)):
              if self.init == 'ortho':
                  init.orthogonal_(module.weight)
              elif self.init == 'N02':
                  init.normal_(module.weight, 0, 0.02)
              elif self.init in ['glorot', 'xavier']:
                  init.xavier_uniform_(module.weight)
              else:
                  print('Init style not recognized...')
              self.param_count += sum([p.data.nelement() for p in module.parameters()])
      print('Param count for D''s initialized parameters: %d' % self.param_count)


    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

        out = self.blocks[-1](h)

        out = out.view(out.size(0),2,self.resolution,self.resolution)
        # out = out.view(out.size(0), 1,self.resolution,self.resolution) # for focal loss 

        # return torch.sigmoid(out) # for bce or focal loss
        return out

# model = Unet_Discriminator(resolution = 256)
# if torch.cuda.is_available():
#   model = model.cuda()

# print('*********************Model Summary***********************')

# print(summary(model, (19,256,256)))


# if __name__ == '__main__': 

#   print('*********************Model Summary***********************')
#   model = UNet_mod(n_class = 2) 
#   if torch.cuda.is_available(): 
#     model = model.cuda()

#   print(summary(model, (19, 256, 256))) 
