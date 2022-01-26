import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
from multiprocessing import cpu_count
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import olympic

import sys
from typing import Union, Callable, Tuple
from functional import boundary, iterated_fgsm, local_search, pgd

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

from livelossplot import PlotLosses


def visualise_adversarial_examples(model, x, y, l2_eps=2, linf_eps=0.15):
    x = x.unsqueeze(0).to(DEVICE)
    y=torch.tensor(y)
    y = y.unsqueeze(0).to(DEVICE)
    x_adv_l2 = pgd(model, x, y, torch.nn.CrossEntropyLoss(), k=120, step=0.1, eps=l2_eps, norm=2)
    x_adv_linf = iterated_fgsm(model, x, y, torch.nn.CrossEntropyLoss(), k=60, step=0.01, eps=linf_eps, norm='inf')
    

    y_pred = model(x)
    y_pred_l2 = model(x_adv_l2)
    y_pred_linf = model(x_adv_linf)
    
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    
    axes[0].imshow(x[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title(
        f'Original sample, '
        f'P({ y_pred.argmax(dim=1).item()}) = '
        f'{np.round(y_pred.softmax(dim=1)[0, y_pred.argmax(dim=1).item()].item(), 3)}')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    axes[1].imshow(x_adv_l2[0, 0].cpu().numpy(), cmap='gray')
    axes[1].set_title(
        f'$L^2$ PGD adversary, '
        f'eps={l2_eps}, '
        f'P({y_pred_l2.argmax(dim=1).item()}) = '
        f'{np.round(y_pred_l2.softmax(dim=1)[0, y_pred_l2.argmax(dim=1).item()].item(), 3)}')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    axes[2].imshow(x_adv_linf[0, 0].cpu().numpy(), cmap='gray')
    axes[2].set_title(
        '$L^{\infty}$ FGSM adversary, '
        f'eps={linf_eps}, '
        f'P({y_pred_linf.argmax(dim=1).item()}) = '
        f'{np.round(y_pred_linf.softmax(dim=1)[0, y_pred_linf.argmax(dim=1).item()].item(), 3)}')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    
    plt.show()

visualise_adversarial_examples(model, *val[19])