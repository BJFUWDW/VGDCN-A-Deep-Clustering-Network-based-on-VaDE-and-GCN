import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np



class load_usps_data(Dataset):
    def __init__(self, dataset):
        # self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        # self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)
        self.x = np.loadtxt('dataset/{}.txt'.format(dataset), dtype=np.float32)
        self.y = np.loadtxt('dataset/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx]))
               # torch.from_numpy(np.array(idx))
# usps_data_loader = load_usps_data('usps')
# usps_data_loader = DataLoader(usps_data_loader, batch_size=256, shuffle=True)
# print(len(usps_data_loader))
# for id,(x,y) in enumerate(usps_data_loader):
   #  print(id)
   #  print(x.size())
