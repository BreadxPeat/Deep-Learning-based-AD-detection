import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim=200):
        super().__init__()
        # encoder
        self.img_2hid1 = nn.Linear(input_dim, 12544)
        self.img_2hid2 = nn.Linear(12544, 2601)
        self.img_2hid3 = nn.Linear(2601, 200)

        # one for mu and one for stds, note how we only output
        # diagonal values of covariance matrix. Here we assume
        # the pixels are conditionally independent
        self.hid_2mu = nn.Linear(200, z_dim)
        self.hid_2sigma = nn.Linear(200, z_dim)

        # decoder
        self.z_2hid1 = nn.Linear(z_dim, 200)
        self.z_2hid2 = nn.Linear(200, 2601)
        self.z_2hid3 = nn.Linear(2601, 12544)
        self.hid_2img = nn.Linear(12544, input_dim)

    def encode(self, x):
        h = F.relu(self.img_2hid1(x))
        h = F.relu(self.img_2hid2(h))
        h = F.relu(self.img_2hid3(h))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        new_h = F.relu(self.z_2hid1(z))
        new_h = F.relu(self.z_2hid2(new_h))
        new_h = F.relu(self.z_2hid3(new_h))
        x = torch.sigmoid(self.hid_2img(new_h))
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)

        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        x = self.decode(z_reparametrized)
        return x, mu, sigma
