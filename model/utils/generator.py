# import necessary modules
import sys
import torch.nn as nn

from .blocks import ResnetBlock, UnetSkipConnectionBlock
from torch.nn import *


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, config):
        """Construct a Resnet-based generator
        Parameters:
            channels_in (int)  -- the number of channels in input images
            channels_out (int) -- the number of channels in output images
            ngf (int)          -- the number of filters in the last conv layer
            norm_layer         -- normalization layer
            dropout (bool)     -- if use dropout layers
            n_layers (int)     -- the number of ResNet blocks
            padding (str)      -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(config["n_layers"] >= 0)
        super(ResnetGenerator, self).__init__()

        # init
        channels_in = config["channels_in"]
        channels_out = config["channels_out"]

        dropout = config["dropout"]
        n_layers = config["n_layers"]
        ngf = config["ngf"]
        padding = config["padding"]
        norm_layer = str_to_class(config["norm"])
        use_bias = config["norm"] != "BatchNorm2d"

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(channels_in,
                           ngf,
                           kernel_size=7,
                           padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult,
                                ngf * mult * 2,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_layers):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult,
                                  padding_type=padding,
                                  norm_layer=norm_layer,
                                  use_dropout=dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult,
                                         int(ngf * mult / 2),
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf,
                            channels_out,
                            kernel_size=7,
                            padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, config):
        """Construct a Unet generator
        Parameters:
            channels_in (int)  -- the number of channels in input images
            channels_out (int) -- the number of channels in output images
            n_layers (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                  image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)          -- the number of filters in the last conv layer
            norm_layer         -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        # init
        channels_in = config["channels_in"]
        channels_out = config["channels_out"]

        dropout = config["dropout"]
        n_layers = config["n_layers"]
        ngf = config["ngf"]
        norm_layer = str_to_class(config["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8,
                                             ngf * 8,
                                             input_nc=None,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(n_layers - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8,
                                                 ngf * 8,
                                                 input_nc=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 use_dropout=dropout)

        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4,
                                             ngf * 8,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2,
                                             ngf * 4,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf,
                                             ngf * 2,
                                             input_nc=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(channels_out,
                                             ngf,
                                             input_nc=channels_in,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
