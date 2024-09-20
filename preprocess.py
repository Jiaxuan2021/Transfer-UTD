from scipy import io, misc
import os
import spectral
import numpy as np
from model.iope_net.iope_net_main import train, test
import torch
import scipy.io as scio
from math import pi, e, cos
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from model.iope_net.model import Encode_Net, Decode_Net1, Decode_Net2


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    # Load Matlab array
    if ext == '.mat':
        return io.loadmat(dataset)
    # Load TIFF file
    elif ext == '.tif' or ext == '.tiff':
        return misc.imread(dataset)
    # Recommend for '.hdr'
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))
    
def get_water_curves(file, have_mask=True):
    """
    file: mat path
    """
    name = os.path.dirname(file).split('/')[-1]
    HSI = open_file(file)['data']   # only for mat file
    HSI_flatten = HSI.reshape((-1, HSI.shape[-1]))
    if have_mask:
        mask_path = f'./dataset/{name}/NDWI_{name}.npy'
        mask = np.load(mask_path)
        water = np.delete(HSI_flatten, np.where(mask.flatten() == 255), axis=0)
        land = np.delete(HSI_flatten, np.where(mask.flatten() == 0), axis=0)
        print(f'Only water shape: {water.shape}, Only land shape: {land.shape}')
        np.save(f'./data_temp/{name}/{name}_water.npy', water)
        np.save(f'./data_temp/{name}/{name}_land.npy', land)
    else:
        print(f'Flatten shape without mask: {HSI_flatten.shape}')
        np.save(f'./data_temp/{name}/{name}_water.npy', HSI_flatten)

def get_IOPs(file, have_mask, device_num):
    """
    Estimation of water quality parameters, call iope-net.
    https://ieeexplore.ieee.org/document/9432747
    """
    name = os.path.dirname(file).split('/')[-1]
    file_list = [f'./data_temp/{name}/{name}_best_model_net0.pth', f'./data_temp/{name}/{name}_best_model_net1.pth', f'./data_temp/{name}/{name}_best_model_net2.pth']
    for files in file_list:
        if not os.path.exists(files):   # There is no trained iope-net model
            get_water_curves(file, have_mask)
            data_path = f'./data_temp/{name}/{name}_water.npy'
            print('Start to train Iope-net, estimation of water quality parameters...')
            train(data_path, device_num)
            print('Iope-net training completed!')
            test(data_path, device_num)
            print('Iope-net prediction completed! Netwrok weights saved in ./data_temp/')
    print('Iope-net weight already exists!')
    

def add_target_pixel(file_name, r_b, r_inf, h, i, j, q, water_num):
    """
    Select the target land pixel, water pixel, and depth 
    to synthesize the reflectance curve of the target underwater

    input:
            r_b: reflectance of target[1d numpy]
            r_inf : reflectanc of optically deep water[1d numpy]
            h: depth
    return:
            synthetic_underwater_target_reflectance: [1d npy]
    param:
            'r'     :   underwater target's off-water reflectance
            'a'     :   absorption coefficient (estimate by the IOPE_Net)
            'bb'     :   scatter coefficient (estimate by the IOPE_Net)
            'k_d'   :   downwelling attenuation coefficient
            'k-uc'  :   upwelling attenuation coefficient of water column
            'k-ub'  :   upwelling attenuation coefficient of target
    """
    # load trained IOPE_Net model, get a,b from 'r_inf'

    bands = r_inf.shape[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net0 = Encode_Net().to(device=device)
    net1 = Decode_Net1().to(device=device)
    net2 = Decode_Net2().to(device=device)
    net0.load_state_dict(torch.load(fr"./data_temp/{file_name}/{file_name}_best_model_net0.pth", map_location=device))
    net1.load_state_dict(torch.load(fr"./data_temp/{file_name}/{file_name}_best_model_net1.pth", map_location=device))
    net2.load_state_dict(torch.load(fr"./data_temp/{file_name}/{file_name}_best_model_net2.pth", map_location=device))
    net0.eval()
    net1.eval()
    net2.eval()

    # copy them for plot, because their format will change to input the net.
    plot_r_inf = r_inf
    plot_r_b = r_b

    r_inf = torch.from_numpy(r_inf).to(device=device, dtype=torch.float32).reshape(1, 1, -1) 
    r_b = torch.from_numpy(r_b).to(device=device, dtype=torch.float32).reshape(1, 1, -1)
    encode_out = net0(r_inf)
    a = net1(encode_out) 
    bb = net2(encode_out)

    # Calculating the reflectance of underwater targets out of water
    u = bb / (a + bb)
    k = a + bb
    k_uc = 1.03 * (1 + 2.4 * u) ** (0.5) * k
    k_ub = 1.04 * (1 + 5.4 * u) ** (0.5) * k
    theta = 0
    k_d = k / cos(theta)
    r = r_inf * (1 - e ** (-(k_d + k_uc) * h)) + r_b  * e ** (-(k_d + k_ub) * h) 
    r = r.squeeze().detach().cpu().numpy() 

    """PLOT"""
    if i==0 and j==water_num-1 and q==100-1:   # The last synthetic spectra
        wavelength = np.linspace(400, 700, bands)
        plt.figure()
        plt.plot(wavelength, gaussian_filter1d(r,sigma=1),
                label='systhetic', color='r', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(plot_r_inf,sigma=1),
                label='water_inf', color='b', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(plot_r_b,sigma=1),
                label='r_b', color='g', marker='o', markersize=3)
        plt.xlabel('wavelength(nm)')
        plt.ylabel('reflectance')
        plt.legend()
        plt.savefig(f'./data_temp/{file_name}/{file_name}_systhetic_curve.png')
    """PLOT"""

    return r

def synthetic_underwater_target(file, have_mask):
    """
    file: mat path
    """
    R_B = scio.loadmat(file)['target']
    file_name = file.split('/')[-2]
    water_curves_path = f'./data_temp/{file_name}/{file_name}_water.npy'
    # R_INF = np.load(water_curves_path)[0:10]    # for simulated dataset 
    R_INF = np.load(water_curves_path)[0:1000]    # for real dataset, used backgroud water pixels, can be adjusted
    water_num = len(R_INF)
    H = np.linspace(0.5, 2, 10)  # range of depth, can be adjusted

    data_len = len(R_B)*len(R_INF)*len(H)
    synthetic_data = np.zeros((data_len, R_B.shape[1]))
    print('Start to synthesize underwater target reflectance...')
    print(f'R_B:{len(R_B)}  R_INF:{len(R_INF)}  H:{len(H)}  number of synthetic data:{data_len}')
    s = 0
    for i in range(len(R_B)):
            for j in range(len(R_INF)):
                    for h in range(len(H)):
                            synthetic_data[s] = add_target_pixel(file_name, R_B[i], R_INF[j], H[h], i, j, h, water_num)
                            s += 1
                            if s%len(H)==0:
                                if j%100==0:
                                    print(f'[{j}/{len(R_INF)}][{i}/{len(R_B)}]')
    target = synthetic_data
    np.save(f'./data_temp/{file_name}/{file_name}_target.npy', synthetic_data)
    if have_mask:
         water = np.load(water_curves_path)
         land = np.load(f'./data_temp/{file_name}/{file_name}_land.npy')
         train_data = np.vstack((water, land, target))
         water_label = np.zeros(len(water)) + 0.05   # soft label
         land_label = np.zeros(len(land))
         target_label = np.ones(len(target)) - 0.05
         train_label = np.hstack((water_label, land_label, target_label))
         print(f'Shape of training data:{train_data.shape}, Shape of training label:{train_label.shape}, with water mask.')
    else:
        water = np.load(water_curves_path)
        train_data = np.vstack((water,target))
        water_label = np.zeros(len(water))
        target_label = np.ones(len(target))
        train_label = np.hstack((water_label, target_label))
        print(f'Shape of training data:{train_data.shape}, Shape of training label:{train_label.shape}, without water mask.')

    return train_data, train_label

