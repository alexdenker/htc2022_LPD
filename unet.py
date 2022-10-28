"""
Adapted from Dival: https://jleuschn.github.io/docs.dival/_modules/dival/reconstructors/networks/unet.html#UNet

+ MaxPool
+ GroupNorm 

author: Alexander Denker
"""


import torch
import torch.nn as nn
import numpy as np





class UNet(nn.Module):

    def __init__(self, in_ch, out_ch, channels, skip_channels, kernel_size = 7, 
                 use_sigmoid=True, use_norm=True, num_groups=8, normalize_input=False):
        super(UNet, self).__init__()
        assert (len(channels) == len(skip_channels))

        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid

        self.use_norm = use_norm
        self.num_groups = num_groups
        self.normalize_input = normalize_input

        if self.normalize_input:
            self.instance_norm = torch.nn.InstanceNorm2d(1)


        if not isinstance(kernel_size, tuple):
            self.kernel_size = [kernel_size]*self.scales 
        else:
            self.kernel_size = kernel_size

        assert (len(channels)) == len(self.kernel_size)

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm, kernel_size=1, num_groups=self.num_groups)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm, 
                                       kernel_size=self.kernel_size[i],
                                       num_groups=self.num_groups))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm, 
                                   kernel_size=self.kernel_size[-i],
                                   num_groups=self.num_groups))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)



    def forward(self, x0):
        if self.normalize_input:
            x0 = self.instance_norm(x0)

        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        return torch.sigmoid(self.outc(x)) if self.use_sigmoid else self.outc(x)




class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True,num_groups=1):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, # one layer more
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, # one layer more
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),)



    def forward(self, x):
        x = self.conv(x)
        return x




class InBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True, num_groups=1):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))



    def forward(self, x):
        x = self.conv(x)
        return x




class UpBlock(nn.Module):

    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, use_norm=True,num_groups=1):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_groups, num_channels=in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))
        if self.skip:
            if use_norm:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                    nn.GroupNorm(num_groups, num_channels=skip_ch),
                    nn.LeakyReLU(0.2, inplace=True))
            else:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.2, inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat()

    def forward(self, x1, x2):
        x = self.up(x1)
        if self.skip:
            x2 = self.skip_conv(x2)
            x = self.concat(x, x2)
        x = self.conv(x)
        return x

class Concat(nn.Module):

    def __init__(self):
        super(Concat, self).__init__()



    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
       # print("CONCAT: ")
       # print("Shapes: ")
       # print("\t From down: ", inputs[0].shape)
       # print("\t From skip: ", inputs[1].shape)

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)




class OutBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)



    def forward(self, x):
        x = self.conv(x)
        return x


    def __len__(self):
        return len(self._modules)
