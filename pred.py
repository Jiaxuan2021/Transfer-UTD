from model.model import *
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
import scipy.io as scio
from utils.cal_AUC_ROC import cal_AUC_ROC
import os
from utils.threshold import threshold

# Without Label
class pred_Loader(Dataset):

    def __init__(self, pred_npy_file):
        self.all_curve = (pred_npy_file).astype(float)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        return curve

    def __len__(self):
        return len(self.all_curve)

def test_net(net, test_dataset, device, batch_size, file):
    gt = scio.loadmat(file)['gt']
    file_name = file.split('/')[-2]
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    net.load_state_dict(torch.load(fr"./data_temp/{file_name}/best_model_{file_name}.pth"))
    net.eval()
    class_map = np.ones((1))
    for curve in test_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        out = net(curve)
        out = out[:, 1]
        out = out.detach().cpu().numpy()
        class_map = np.hstack((class_map, out))
    class_map = np.delete(class_map, 0, 0)
    data = scio.loadmat(file)['data']
    r, c, b = data.shape
    class_map = np.resize(class_map, (r, c))
    # AUC-ROC
    auc = cal_AUC_ROC(class_map, gt, file_name, is_show=True)
    print(fr'The detection AUC value is: {auc:.4f}')
    np.save(fr'./data_temp/{file_name}/result_{file_name}.npy', class_map)  
    matplotlib.image.imsave(fr'./data_temp/{file_name}/result_{file_name}.png', class_map)  

    threshold_map = threshold(class_map, threshold=0.1)
    matplotlib.image.imsave(fr'./data_temp/{file_name}/threshold_{file_name}.png', threshold_map)


if __name__ == "__main__":

    file = f'./dataset/ningxiang/data.mat'
    device_num = '0'

    HSI = scio.loadmat(file)['data']
    HSI_flatten = HSI.reshape((-1, HSI.shape[-1]))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ResNet = ResNet_UTD().to(device=device)
    dataset = pred_Loader(HSI_flatten)
    test_net(net=ResNet, test_dataset=dataset, device=device, batch_size=4000, file=file)