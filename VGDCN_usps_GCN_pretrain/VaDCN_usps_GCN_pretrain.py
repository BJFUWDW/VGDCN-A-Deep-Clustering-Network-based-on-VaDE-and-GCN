import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
from makedataset import load_usps_data
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from construction_graph import load_graph

usps_data_loader = load_usps_data('usps')

class VAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, hid_dim=10):
        super(VAE, self).__init__()

        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)

        self.mu_l = nn.Linear(n_enc_3, hid_dim)
        self.log_sigma2_l = nn.Linear(n_enc_3, hid_dim)

        self.dec_1 = nn.Linear(hid_dim, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        tra1 = F.relu(self.enc_1(x))
        tra2 = F.relu(self.enc_2(tra1))
        tra3 = F.relu(self.enc_3(tra2))
        mu = self.mu_l(tra3)
        logvar_sigma2 = self.log_sigma2_l(tra3)
        z = torch.randn_like(mu) * torch.exp(logvar_sigma2 / 2) + mu

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return tra1, tra2, tra3, mu, logvar_sigma2, z, x_bar

def pretrain_vae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        for batch_id, (x, _) in enumerate(train_loader):
            x = x.cuda()

            _, _, _, _, _, _, x_bar = model.forward(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            _, _, _, _, _, _, x_bar = model.forward(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))

        model.log_sigma2_l.load_state_dict(model.mu_l.state_dict())
        torch.save(model.state_dict(), 'usps.pkl')

model = VAE(
    500, 500, 2000, 2000, 500, 500, 256, hid_dim=10
)
model = model.cuda()
pretrain_vae(model, usps_data_loader, usps_data_loader.y)
