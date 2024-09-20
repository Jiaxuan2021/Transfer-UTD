from model.iope_net.model import Encode_Net, Decode_Net1, Decode_Net2
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch
import sys
import numpy as np
import math
import torch.nn.functional as F
import os


class HSI_Loader(Dataset):

    def __init__(self, file):
        # load dataset
        self.all_curve = np.load(file)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        # label is curve itself
        label = torch.tensor(self.all_curve[index])
        return curve, label

    def __len__(self):
        return len(self.all_curve)


def train_net(net0, net1, net2, device, file, epochs, batch_size, lr):
    # Load dataset
    data_name = file.split('/')[-1].split('.')[0].split('_water')[0]
    HSI_dataset = HSI_Loader(file)
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset, batch_size=batch_size, shuffle=True)
    # 3 nets use the same RMSprop and learning rate
    optimizer0 = optim.RMSprop(net0.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer1 = optim.RMSprop(net1.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    optimizer2 = optim.RMSprop(net2.parameters(), lr=lr, weight_decay=0, momentum=0.9)
    # Best_loss, start with 'inf'
    best_loss = float('inf')
    # Train
    for epoch in range(epochs):
        # Train mode
        net0.train()
        net1.train()
        net2.train()
        for curve, label in train_loader:
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # Load data and label to device, curve add 1 dim so that it can feed into the net
            curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            encode_out = net0(curve)
            a = net1(encode_out)
            bb = net2(encode_out)
            # Bathymetric model-based reconstruction
            u= bb / (a + bb)
            r = (0.084 + 0.170 * u) * u
            r = torch.squeeze(r)
            # Here, r and label both [1024,176]
            # Define MSE Loss
            MSE_Loss = nn.MSELoss()
            # Define SA(Spectral Angle) Loss
            def SA_Loss(r, label):
                r_l2_norm = torch.norm(r, p=2, dim=1) # [1024]
                label_l2_norm = torch.norm(label, p=2, dim=1) # [1024]
                SALoss =  torch.sum(torch.acos(torch.sum(r*label, dim=1)/(r_l2_norm*label_l2_norm)))
                SALoss /= math.pi * len(r)
                return SALoss
            # First, train using only MSE and determine 1e7 according to the convergence magnitude
            loss = 1e7*MSE_Loss(r, label) + SA_Loss(r, label)
            # Save the model parameters with the lowest loss
            if loss < best_loss:
                best_loss = loss
                torch.save(net0.state_dict(), f'./data_temp/{data_name}/{data_name}_best_model_net0.pth')
                torch.save(net1.state_dict(), f'./data_temp/{data_name}/{data_name}_best_model_net1.pth')
                torch.save(net2.state_dict(), f'./data_temp/{data_name}/{data_name}_best_model_net2.pth')
            # Back propagation, update parameters
            loss.backward()
            optimizer0.step()
            optimizer1.step()
            optimizer2.step()
        if epoch % 50 == 0:
            print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


def train(file, device_num):
    """
    File: Water curve data
    """
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    # Load nets
    net0 = Encode_Net()
    net1 = Decode_Net1()
    net2 = Decode_Net2()
    # Load  nets to device
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    # Give dataset, start train!
    train_net(net0, net1, net2, device, file, epochs=500, batch_size=1024, lr=0.00001)  # 500 epoch


def pred_net(net0, net1, net2, device, file, batch_size):
    """
    File: Water curve data
    """
    data_name = file.split('/')[-1].split('.')[0].split('_water')[0]
    HSI_dataset = HSI_Loader(file)
    bands = np.load(file).shape[-1]
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset, batch_size=batch_size, shuffle=False)
    # Load the trained network model parameters
    net0.load_state_dict(torch.load(f'./data_temp/{data_name}/{data_name}_best_model_net0.pth', map_location=device))
    net1.load_state_dict(torch.load(f'./data_temp/{data_name}/{data_name}_best_model_net1.pth', map_location=device))
    net2.load_state_dict(torch.load(f'./data_temp/{data_name}/{data_name}_best_model_net2.pth', map_location=device))
    # Pred mode
    net0.eval()
    net1.eval()
    net2.eval()
    for curve, label in train_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        encode_out = net0(curve)
        a = net1(encode_out)
        bb = net2(encode_out)
        u = bb / (a+bb)
        r = (0.084 + 0.170 * u) * u
        r = torch.squeeze(r)
        # select a curve to plot
        pltcurve = 100
        print(f'r:{r.shape}')
        print(f'a:{a.shape}')
        print(f'bb:{bb.shape}')
        print(f'label:{label.shape}')
        MSE_Loss = nn.MSELoss()
        def SA_Loss(r, label):
            r_l2_norm = torch.norm(r, p=2, dim=1)  # [1024]
            label_l2_norm = torch.norm(label, p=2, dim=1)  # [1024]
            SALoss = torch.sum(torch.acos(torch.sum(r * label, dim=1) / (r_l2_norm * label_l2_norm)))
            SALoss /= math.pi * len(r)
            return SALoss
        loss = 1e7 * MSE_Loss(r, label) + SA_Loss(r, label)
        # REP(relative error percentage)
        REP = torch.abs(r-label)/label # REP [1024,176]
        sum_REP = torch.where(torch.isinf(torch.sum(REP, 1)), torch.full_like(torch.sum(REP, 1), 1), torch.sum(REP, 1)) # some 'inf' change to '1'
        mean_REP = torch.mean(sum_REP)
        print(f'Loss:{loss.item()}')
        print(f'REP:{torch.sum(REP[pltcurve,:]).item()}')
        print(f'mean_REP:{mean_REP.item()}')

        # Plot real(label) and reconstruct curve
        wavelength = np.linspace(400, 1000, bands)
        plt.figure()
        # Gaussian filtering can be removed
        plt.plot(wavelength, gaussian_filter1d(torch.squeeze(r[pltcurve,:]).detach().cpu().numpy(),sigma=5),
                 label='Reconstruct', color='r', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(torch.squeeze(label[pltcurve,:]).detach().cpu().numpy(),sigma=5),
                 label='Real', color='b', marker='o', markersize=3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance Value')
        plt.legend()
        plt.savefig(f'./data_temp/{data_name}/{data_name}_real_reconstruct_curve.png')
        plt.clf()
        # Plot estimated 'a', 'bb'
        plt.figure()
        a_smoothed = gaussian_filter1d(torch.squeeze(a[pltcurve,:]).detach().cpu().numpy(), sigma=5)
        plt.plot(wavelength, a_smoothed,
                 label='a', color='r', marker='o', markersize=3)
        b_smoothed = gaussian_filter1d(torch.squeeze(bb[pltcurve,:]).detach().cpu().numpy(), sigma=5)
        plt.plot(wavelength, b_smoothed,
                 label='bb', color='b', marker='o', markersize=3)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('m^(-1)')
        plt.legend()
        plt.savefig(f'./data_temp/{data_name}/{data_name}_estimated_a_bb.png')
        plt.clf()
        break

def test(file, device_num):
    """
    File: Water curve data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    net0 = Encode_Net()
    net1 = Decode_Net1()
    net2 = Decode_Net2()
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    pred_net(net0, net1, net2, device, file, batch_size=1024)