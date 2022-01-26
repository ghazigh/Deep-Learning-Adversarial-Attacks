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


transform = transforms.Compose([transforms.ToTensor(),])

train = datasets.MNIST('../data/', train=True, transform=transform, download=True)
val = datasets.MNIST('../data/', train=False, transform=transform, download=True)

train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
val_loader = DataLoader(val, batch_size=128, num_workers=cpu_count())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, num_epochs=10):
    liveloss = PlotLosses()
    model = model.to(device)
    
    for epoch in range(num_epochs):
        logs = {}
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
        
        liveloss.update(logs)
        liveloss.send()

model = Net().to(DEVICE)
optimiser = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

criterion = loss_fn
optimizer = optimiser

dataloaders = {
    "train": train_loader,
    "validation": val_loader
}

#train_model(model, criterion, optimizer, num_epochs=20)

history_natural = olympic.fit(
    model,
    optimiser,
    loss_fn,
    dataloader=train_loader,
    epochs=10,
    metrics=['accuracy'],
    prepare_batch = lambda batch: (batch[0].to(DEVICE), batch[1].to(DEVICE)),
    callbacks=[
        olympic.callbacks.Evaluate(val_loader),
        olympic.callbacks.ReduceLROnPlateau(patience=5)
    ]
)

torch.save(model, 'mnist_model.pt')