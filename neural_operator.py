# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import os
import sys
import pickle

from utilities3 import *
from sklearn.model_selection import train_test_split
from Adam import Adam
from timeit import default_timer

# %%
# read data from csv file
gc = pd.read_csv("Data/gc_samples_filtered.csv", header=None)
d = pd.read_csv("Data/d_samples_filtered.csv", header=None)

# %%
# check number of samples in gc
print("Number of samples in gc: ", len(gc))

# %%
# coordinates data from csv file
coordinates = pd.read_csv("Data/coordinates_n", header=None)

# %%
#  convert to torch tensor
print("gc shape: ", gc.shape)
print("d shape: ", d.shape)
print("coordinates shape: ", coordinates.shape)

# %%
# create numpy array with shape (len(gc), gc.shape[1], 2)
input_mesh = np.zeros((len(gc), gc.shape[1], 2))

# for each sample in input_mesh, add the coordinates
for i in range(len(gc)):
    input_mesh[i, :, :] = coordinates
    
# add gc to the last dimension of input_mesh to have shape of (len(gc), gc.shape[1], 3)
input_data = np.concatenate((input_mesh, np.expand_dims(gc, axis=2)), axis=2)
input_data = torch.from_numpy(input_data).float()

d = torch.from_numpy(d.values).float()


# %%
# train test split X data, for use in creating TensorDataset
X_train, X_test, y_train, y_test = train_test_split(input_data, d, test_size=0.2, random_state=42)

batch_size = 32;
train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


# %%
# import local modules from FNO2D.py
from FNO2D import *

# %%
# config for model

batch_size = 32
learning_rate = 0.001

epochs = 501
step_size = 50
gamma = 0.5

modes = 12
width = 40

# %%
model = FNO2d(modes, modes, width=width, in_channels=3, out_channels=1).cuda()
model_iphi = IPHI(width=width).cuda()

# %%
print(count_params(model), count_params(model_iphi))

params = list(model.parameters()) + list(model_iphi.parameters())
optimizer = Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# %%
X_train.shape[0]

# %%
myloss = LpLoss(size_average=False)
N_sample = 1000 # number of samples for regularization
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for data, damage in train_loader:
        damage, data = damage.cuda(), data.cuda()
        samples_x = torch.rand(batch_size, N_sample, 3).cuda() * 3 -1

        optimizer.zero_grad()
        out = model(data, iphi=model_iphi)
        samples_xi = model_iphi(samples_x)

        loss_data = myloss(out.view(batch_size, -1), damage.view(batch_size, -1))
        loss_reg = myloss(samples_xi, samples_x)
        loss = loss_data + 0.000 * loss_reg
        loss.backward()

        optimizer.step()
        train_l2 += loss_data.item()
        train_reg += loss_reg.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, damage, data in test_loader:
            rr, damage, data = rr.cuda(), damage.cuda(), data.cuda()
            # out = model(mesh, iphi=model_iphi)
            out = model(data, code=rr, iphi=model_iphi)
            test_l2 += myloss(out.view(batch_size, -1), damage.view(batch_size, -1)).item()

    train_l2 /= X_train.shape[0]
    train_reg /= X_train.shape[0]
    test_l2 /= X_test.shape[0]

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, train_reg, test_l2)

    if ep%100==0:
        XY = data[-1].squeeze().detach().cpu().numpy()
        truth = damage[-1].squeeze().detach().cpu().numpy()
        pred = out[-1].squeeze().detach().cpu().numpy()

        # lims = dict(cmap='RdBu_r', vmin=truth.min(), vmax=truth.max())
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        # ax[0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
        # ax[1].scatter(XY[:, 0], XY[:, 1], 100, pred, edgecolor='w', lw=0.1, **lims)
        # ax[2].scatter(XY[:, 0], XY[:, 1], 100, truth - pred, edgecolor='w', lw=0.1, **lims)
        # fig.show()


# %%



