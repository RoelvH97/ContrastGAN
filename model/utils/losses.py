# import necessary modules
import torch
import torch.nn as nn

from torch import Tensor


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


class NCCLoss(nn.Module):
    def __init__(self, config):
        super(NCCLoss, self).__init__()
        self.e = config["e"]
        self.stablestd = StableStd.apply

    def ncc(self, x1, x2):
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = self.stablestd(x1) * self.stablestd(x2)
        ncc = cc / (std + self.e)
        return ncc

    def forward(self, input, target):
        if isinstance(input, list):
            reconstruction, mu, log_var = input[0], input[1], input[2]
        else:
            reconstruction = input

        return -self.ncc(reconstruction, target)


class HULoss(nn.Module):
    def __init__(self, config):
        super(HULoss, self).__init__()
        self.bias = config["bias"]
        self.factor = config["factor"]

        self.min = (config["min"] + config["bias"]) / config["factor"]
        self.max = (config["max"] + config["bias"]) / config["factor"]

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        input_masked = torch.masked_select(input, mask)

        loss_min = torch.mean((torch.minimum(input_masked, torch.tensor(self.min)) - self.min) ** 2)
        loss_max = torch.mean((torch.maximum(input_masked, torch.tensor(self.max)) - self.max) ** 2)
        return loss_min + loss_max


class WassersteinLoss(nn.Module):
    @staticmethod
    def forward(input, target=None) -> Tensor:
        if target:
            return torch.mean(input) - torch.mean(target)
        else:
            return torch.mean(input)
