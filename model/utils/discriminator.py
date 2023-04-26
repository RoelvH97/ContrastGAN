# import necessary modules
import sys
import torch.nn as nn

from torch.nn import *


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, config):
        super(NLayerDiscriminator, self).__init__()

        # init
        channels_in = config["channels_in"]
        channels_out = config["channels_out"]

        n_layers = config["n_layers"]
        ndf = config["ndf"]
        norm_layer = str_to_class(config["norm"])
        use_bias = config["norm"] != "BatchNorm2d"

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(channels_in, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev,
                      ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, channels_out, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
