import scipy.io as sio
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def cal_AUC_ROC(result, gt, file_name, is_show):
    if (np.max(result) - np.min(result)) == 0:
        return 0
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    gt = gt.flatten()
    result = result.flatten()
    fpr, tpr, _ = roc_curve(gt, result)
    roc_auc = auc(fpr, tpr)

    # plot ROC
    if is_show:
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower right")
        plt.savefig(f'./data_temp/{file_name}/ROC_{file_name}.png')
        plt.clf()

    return roc_auc