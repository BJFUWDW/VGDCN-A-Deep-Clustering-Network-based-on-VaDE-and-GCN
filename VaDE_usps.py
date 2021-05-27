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
from scipy.interpolate import make_interp_spline

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


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]
    return layers


class Encoder(nn.Module):
    def __init__(self,input_dim=256, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l=nn.Linear(inter_dims[-1],hid_dim)
        self.log_sigma2_l=nn.Linear(inter_dims[-1],hid_dim)

    def forward(self, x):
        e=self.encoder(x)
        # print(e.size())
        mu=self.mu_l(e)
        log_sigma2=self.log_sigma2_l(e)

        return mu,log_sigma2


class Decoder(nn.Module):
    def __init__(self,input_dim=256,inter_dims=[500, 500, 2000], hid_dim=10):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(hid_dim,inter_dims[-1]),
            *block(inter_dims[-1],inter_dims[-2]),
            *block(inter_dims[-2],inter_dims[-3]),
            nn.Linear(inter_dims[-3],input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_pro=self.decoder(z)

        return x_pro


class VaDE(nn.Module):
    def __init__(self, nClusters, hid_dim):
        super(VaDE,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.pi_ = nn.Parameter(torch.FloatTensor(nClusters,).fill_(1)/nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(nClusters, hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(nClusters, hid_dim).fill_(0), requires_grad=True)

        self.nClusters = nClusters
        self.hid_dim = hid_dim
    def pre_train(self, dataloader, pre_epoch=20):

        if not os.path.exists('./pretrain_model.pk'):

            Loss = nn.MSELoss()
            opti = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar = tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L = 0
                for x, y in dataloader:
                    x = x.view(x.size(0), -1)
                    x = Variable(x)
                    x = x.cuda()
                    z, _ = self.encoder(x)
                    x_ = self.decoder(z)
                    loss = Loss(x, x_)

                    L += loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L / len(dataloader)))

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            Z = []
            Y = []
            with torch.no_grad():
                for x, y in dataloader:

                    x = x.view(x.size(0), -1)
                    x = Variable(x)
                    x = x.cuda()

                    z1, z2 = self.encoder(x)

                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()
            print(Z.size)
            print(Y.size)
            gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
            mu_c = self.mu_c.data
            torch.save(self.state_dict(), './pretrain_model.pk')

        else:

            self.load_state_dict(torch.load('./pretrain_model.pk'))

    def predict(self, x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def ELBO_Loss(self, x, y, L=1):
        det = 1e-10

        L_rec = 0

        z_mu, z_sigma2_log = self.encoder(x)

        # print(z_mu.size())
        #plt.scatter(z_mu.detach().cpu().numpy(), z_sigma2_log.detach().cpu().numpy(), c=y.detach().cpu().numpy())
        #plt.show()
        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu

            x_pro = self.decoder(z)
            # gen_images = x_pro.view(-1, 1, 16, 16)
            # save_image(gen_images, 'E:/!!!!毕业设计/资料/CODES/!!My_DCN/USPS_decoder/usps_decode-{}{}.png'.format(epoch + 1, batch_id + 1))

            # print(x_pro.size())
            L_rec += F.binary_cross_entropy(x_pro, x)

        L_rec /= L

        Loss = L_rec * x.size(1)

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
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

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))


# mnist_data_loader = mnist_dataset('./mnist/')
usps_data_loader = load_usps_data('usps')
usps_data_loader = DataLoader(usps_data_loader, batch_size=258, shuffle=True)


vade = VaDE(10, 10)
vade = vade.cuda()
vade.pre_train(usps_data_loader, pre_epoch=20)
optimizer = Adam(vade.parameters(), lr=1e-3)
lr_s = StepLR(optimizer,step_size=10,gamma=0.95)
writer = SummaryWriter('./logs')

epoch_bar = tqdm(range(400))

tsne = TSNE()

for epoch in epoch_bar:

    # lr_s.step()
    L = 0
    for batch_id, (x, y) in enumerate(usps_data_loader):

        x = x.view(x.size(0), -1)
        x = Variable(x)
        x = x.cuda()
        loss = vade.ELBO_Loss(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        L += loss.detach().cpu().numpy()

    lr_s.step()
    pre = []
    tru = []
    mu_sum = []
    logvar_sum = []
    with torch.no_grad():
        for batch_id,(x, y) in enumerate(usps_data_loader):


            x = x.view(x.size(0), -1)
            x = Variable(x)
            x = x.cuda()
            tru.append(y.numpy())
            pre.append(vade.predict(x))

    tru = np.concatenate(tru, 0)
    pre = np.concatenate(pre, 0)

    writer.add_scalar('loss', L / len(usps_data_loader), epoch)
    writer.add_scalar('acc', cluster_acc(pre, tru)[0] * 100, epoch)
    writer.add_scalar('lr', lr_s.get_last_lr()[0], epoch)

    epoch_bar.write('Loss={:.4f},ACC={:.4f}%,LR={:.4f}'.format(L / len(usps_data_loader), cluster_acc(pre, tru)[0] * 100, lr_s.get_last_lr()[0]))



