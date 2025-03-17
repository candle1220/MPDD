from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_channels: int,
                 latent_dim: int,) -> None:
        super().__init__()
        self.proc = nn.Linear(in_channels,latent_dim)
        self.batch1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.LeakyReLU()
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_logit = nn.Linear(latent_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, in_channels)

        self.res = nn.Linear(latent_dim, in_channels)
        self.batch2 = nn.BatchNorm1d(in_channels)
        self.sig = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, support: Tensor) -> List[Tensor]:
        batch_size = support.size(0)
        x = self.proc(support)
        if not batch_size==1:
            x = self.batch1(x)
        result = self.relu(x)
        means_feat = self.fc_mean(result)
        logit_feat = self.fc_logit(result)

        std = torch.exp(0.5 * logit_feat)
        eps = torch.randn_like(std)
        feat = eps * std + means_feat
        uncouple_info = std + means_feat
        z_output = self.decoder_input(uncouple_info)

        distance = 1 + logit_feat - means_feat ** 2 - logit_feat.exp()

        y = self.res(feat)
        if not batch_size==1:
            y = self.batch2(y)
        new_feat = self.sig(y)

        return [new_feat, z_output, distance]


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        validity = torch.sigmoid(self.fc2(z))
        return validity