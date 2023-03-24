import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RateDistortionLoss(nn.Module):
    def __init__(self,lmbda):
        super().__init__()
        self.lmbda = lmbda

    def forward(self,x,x_hat,y_likelihoods,z_likelihoods):
        # bitrate of the quantized latent
        N, _, H, W = x.size()
        num_pixels = N * H * W
        bpp_loss_y = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
        bpp_loss_z = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

        # mean square error
        mse_loss = F.mse_loss(x, x_hat)

        # final loss term
        loss = mse_loss + self.lmbda * bpp_loss_y + self.lmbda * bpp_loss_z
        return loss