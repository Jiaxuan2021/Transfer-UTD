from model.model import *
from torch.utils.data import Dataset
from torch import optim
import torch
from preprocess import *
from utils.cal_AUC_ROC import cal_AUC_ROC


class train_Loader(Dataset):

    def __init__(self, train_npy_file, label_npy_file):
        self.all_curve = (train_npy_file).astype(float)
        self.all_label = (label_npy_file).astype(float)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        label = torch.tensor(self.all_label[index])
        return curve, label

    def __len__(self):
        return len(self.all_curve)
    
class val_Loader(Dataset):

    def __init__(self, pred_npy_file):
        self.all_curve = (pred_npy_file).astype(float)

    def __getitem__(self, index):
        curve = torch.tensor(self.all_curve[index])
        return curve

    def __len__(self):
        return len(self.all_curve)
    

def train_net(net, train_dataset, val_dataset, device, batch_size, lr, epochs, file_name, hsi, gt):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=4000, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, eps=1e-08,  weight_decay=1e-2,  momentum=0.9)
    best_loss = float('inf')
    best_epoch_auc = 0

    for epoch in range(epochs):
        net.train()
        for curve, label in train_loader:
            optimizer.zero_grad()
            curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)     
            out = net(curve)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            net.eval()
            class_map = np.ones((1))
            for curve in validate_loader:
                curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
                out = net(curve)
                out = out[:, 1]
                out = out.detach().cpu().numpy()
                class_map = np.hstack((class_map, out))
            class_map = np.delete(class_map, 0, 0)
            data = hsi
            r, c, b = data.shape
            class_map = np.resize(class_map, (r, c))
            auc = cal_AUC_ROC(class_map, gt, file_name, is_show=False)
            if auc > best_epoch_auc and epoch > 2:
                best_epoch_auc = auc
                torch.save(net.state_dict(), fr"./data_temp/{file_name}/best_model_{file_name}.pth")
                # np.save(fr"./data_temp/{file_name}/TransferUTD_{file_name}_{auc}.npy", class_map)
        print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}, auc:{auc:.4f} ,Best auc:{best_epoch_auc:.4f}')

if __name__ == '__main__':

    data_path = f'./dataset/River_scene2/data.mat'   # The path of the dataset
    device_num = '0' 
    is_have_mask = True    # If the dataset has NDWI mask, set it to True, you can generate the mask by yourself

    print(data_path)

    data = scio.loadmat(data_path)['data']
    hsi_flatten = data.reshape((-1, data.shape[-1]))
    gt = scio.loadmat(data_path)['gt']
    file_name = data_path.split('/')[-2]

    if os.path.exists(f'./data_temp/{file_name}') == False:
        os.makedirs(f'./data_temp/{file_name}')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on GPU: ' + device_num)
    get_IOPs(data_path, is_have_mask, device_num=device_num)
    print('IOPs estimation completed!')
    train_data, train_label = synthetic_underwater_target(data_path, is_have_mask)
    ResNet = ResNet_UTD().to(device=device)
    dataset = train_Loader(train_data, train_label)
    val_dataset = val_Loader(hsi_flatten)
    print('Start to train TransferUTD...')
    train_net(net=ResNet, train_dataset=dataset, val_dataset=val_dataset, device=device, batch_size=512, lr=1e-5, epochs=300, file_name=file_name, hsi=data, gt=gt)  # 300 epoch
    print('TransferUTD training completed!')