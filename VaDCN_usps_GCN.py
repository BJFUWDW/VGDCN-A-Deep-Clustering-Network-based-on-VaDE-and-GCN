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
from makedataset import train_dataset, test_dataset, pre_train_dataset, mnist_dataset, load_usps_data
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from construction_graph import load_graph


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    # print(Y_pred)
    # print(Y)
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    right = np.amax(w, axis=1)
    right = sum(right)
    # print(w)
    # print(w.max())
    # ind = linear_assignment(w.max() - w)
    # print(ind)
    return right/Y_pred.size, w


class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


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
        self.sigmoid = nn.Sigmoid()

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
        x_pro = self.x_bar_layer(dec_h3)
        x_bar = self.sigmoid(x_pro)


        return tra1, tra2, tra3, mu, logvar_sigma2, z, x_bar


class VaDE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, hid_dim=10, nClusters=10):
        super(VaDE,self).__init__()
        self.vae = VAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            hid_dim=hid_dim
        )
        self.vae.load_state_dict(torch.load('VaDCN_GCN_pretrain/usps.pkl', map_location='cpu'))
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, hid_dim)
        self.gnn_5 = GNNLayer(hid_dim, nClusters)

        self.pi_ = nn.Parameter(torch.FloatTensor(nClusters,).fill_(1)/nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(nClusters, hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(nClusters, hid_dim).fill_(0), requires_grad=True)

        self.v = 1
        self.nClusters = nClusters
        self.hid_dim = hid_dim

    def forward(self, x, adj):
        #VAE Module
        tra1, tra2, tra3, mu, logvar_sigma2, z, x_bar = self.vae(x)

        sigma = 0.5

        #GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        predict = F.softmax(h, dim=1)

        #Evaluate q, p
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.mu_c, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return mu, logvar_sigma2, z, x_bar, predict, q



    def predict_vae(self, x):
        _, _, _, z_mu, z_sigma2_log, _, _ = self.vae(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))
        #print(yita_c.size())
        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)


    def predict_GCN(self,x ,adj):

        _, _, _, _, predict, _ = self.forward(x, adj)
        predict = predict.detach().cpu().numpy()
        return np.argmax(predict, axis=1)


    def ELBO_Loss(self, x):
        det = 1e-10

        L_rec = 0

        _, _, _, z_mu, z_sigma2_log, z, x_pro = self.vae(x)

        L_rec += F.binary_cross_entropy(x_pro, x)

        Loss = L_rec * x.size(1)

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

        Loss += 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                              torch.exp(
                                                                  z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(
                                                                      0)) +
                                                              (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(
                                                                  2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z_sigma2_log, 1))

        return Loss

    def GCN_Loss(self, x, adj):
        mu, logvar_sigma2, z, x_bar, predict, q = self.forward(x, adj)
        q = q.data
        p = target_distribution(q)
        delta = 0.01
        GCN_Loss = F.kl_div(predict.log(), p, reduction='batchmean')
        GCN_Loss = delta*GCN_Loss
        return GCN_Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# load data and graph

usps_data_loader = load_usps_data('usps')
adj = load_graph('usps', 10)
adj = adj.cuda()

model = VaDE(500, 500, 2000, 2000, 500, 500, 256, hid_dim=10, nClusters=10)
model = model.cuda()
optimizer = Adam(model.parameters(), lr=2e-3)
lr_s = StepLR(optimizer,step_size=10,gamma=0.95)
writer = SummaryWriter('./logs')
#epoch_bar = tqdm(range(400))
tsne = TSNE()

# cluster parameter initiate
x = torch.Tensor(usps_data_loader.x).cuda()
y = usps_data_loader.y
with torch.no_grad():
    _, _, _, z_mu, z_sigma2_log, z, x_pro = model.vae(x)
gmm = GaussianMixture(n_components=model.nClusters, covariance_type='diag')
pre = gmm.fit_predict(z.detach().cpu().numpy())
print('Acc={:.4f}%'.format(cluster_acc(pre, y)[0] * 100))
model.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
model.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
model.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

#train
for epoch in range(200):
    L = 0
    loss = model.ELBO_Loss(x) + model.GCN_Loss(x, adj)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    L += loss.detach().cpu().numpy()

    lr_s.step()
    VaDE_pre = []
    GCN_pre = []
    tru = []
    with torch.no_grad():
        tru.append(y)
        VaDE_pre.append(model.predict_vae(x))
        GCN_pre.append(model.predict_GCN(x, adj))
    tru = np.concatenate(tru, 0)
    VaDE_pre = np.concatenate(VaDE_pre, 0)
    GCN_pre = np.concatenate(GCN_pre, 0)

    # writer.add_scalar('loss', L / len(usps_data_loader), epoch)
    # writer.add_scalar('VaDE_acc', cluster_acc(VaDE_pre, tru)[0] * 100, epoch)
    # writer.add_scalar('GCN_acc', cluster_acc(GCN_pre, tru)[0] * 100, epoch)
    # writer.add_scalar('lr', lr_s.get_last_lr()[0], epoch)
    #
    # epoch_bar.write(
    #     'Loss={:.4f},VaDE_ACC={:.4f}%,GCN_ACC={:.4f}%,LR={:.4f}'.format(L / len(usps_data_loader), cluster_acc(VaDE_pre, tru)[0] * 100, cluster_acc(GCN_pre, tru)[0] * 100, lr_s.get_last_lr()[0]))

    print('{:.4f},'.format(cluster_acc(GCN_pre, tru)[0] * 100))
    #print('{:.4f},'.format(cluster_acc(VaDE_pre, tru)[0] * 100))