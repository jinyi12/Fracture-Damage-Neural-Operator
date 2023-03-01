# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dadaptation
import random


from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import h5py
import os
import sys
import pickle
import json

from utilities3 import *
from sklearn.model_selection import train_test_split
from Adam import Adam
from timeit import default_timer
from collections import OrderedDict

# import local modules from FNO2D.py
import FNO2D


import wandb
import datetime

# %%
# get device
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

PROJECT_NAME = 'FNO2D'

# Set the random seeds to improve reproducibility by removing stochasticity
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False # Force cuDNN to use a consistent convolution algorithm
    torch.backends.cudnn.deterministic = True # Force cuDNN to use deterministic algorithms if available
    torch.use_deterministic_algorithms(True) # Force torch to use deterministic algorithms if available

set_seeds(0)

# for deterministic pytorch algorithms, enable reproducibility.
os.environ['CUBLAS_WORKSPACE_CONFIG']= ":4096:8"

# %%
# with wandb.init(project="FNO2D", entity="jyyresearch", job_type="get-raw-data") as run:
#     raw_data = run.use_artifact('jyyresearch/FNO2D/fracture-damage-raw-data:latest', type='raw_data')
#     raw_data_dir = raw_data.download()
#     gc_data = h5py.File(raw_data_dir + "/gc_data", "r")
#     damage_data = h5py.File(raw_data_dir + "/damage_data", "r")

# %%
config = {
    'train_val_split': [0.80, 0.20], # These must sum to 1.0
    'batch_size' : 16, # Num samples to average over for gradient updates
    'EPOCHS' : 200, # Num times to iterate over the entire dataset
    'LEARNING_RATE' : 1e-3, # Learning rate for the optimizer
    'BETA1' : 0.9, # Beta1 parameter for the Adam optimizer
    'BETA2' : 0.999, # Beta2 parameter for the Adam optimizer
    'WEIGHT_DECAY' : 1e-4, # Weight decay parameter for the Adam optimizer
    'accum_iter': 16, # iterations to accumulate gradients
}


# %% [markdown]
# ### define some helper functions for transforming numpy to tensors

# %%
class ToTensor(object):
    """Convert numpy arrays to tensor arrays
    """
    def __init__(self, device=None):
        if device is None:
            device = "cpu"
        self.device = device
    
    def __call__(self, data):
        if self.device == "cpu":
            return torch.from_numpy(data)
        else:
            # to overlap data transfers with computation, use non_blocking=True
            return torch.from_numpy(data).to(self.device, non_blocking=True, dtype=torch.float32)

# %%
def get_transforms(transform_dict):
    """
    Given a dictionary of transform parameters, return a list of class instances for each transform
    Arguments:
        transform_dict (OrderedDict) with optional keys:
            ToTensor (dict) if present, requires the 'device' key that indicates the PyTorch device
    Returns:
        composed_transforms (PyTorch composed transform class) containing the requested transform steps in order
    """
    transform_functions = []
    for key in transform_dict.keys():
        if key=='ToTensor': # Convert array to a PyTorch Tensor
            transform_functions.append(ToTensor(
                transform_dict[key]['device']
            ))
        
    composed_transforms = transforms.Compose(transform_functions)
    return composed_transforms

# %%
# create a torch dataset
class FractureDamageDataset(torch.utils.data.Dataset):
    def __init__(self, gc_data, damage_data, transform=None):
        self.gc_data = gc_data
        self.damage_data = damage_data
        self.transform = transform

    def __len__(self):
        return len(self.gc_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        gc = self.gc_data[idx] 
        damage = self.damage_data[idx]
        if self.transform:
            gc = self.transform(gc)
            damage = self.transform(damage)
        return gc, damage

# %%
def make_split_artifact(run, train_rows, val_rows):
    """
    Creates a w&b artifact that contains the train and validation rows of the raw data
        run (wandb run) returned from wandb.init()
        train_rows (list of ints) indices that reference the training rows in the raw_data
        val_rows (list of ints) indices that reference the validation rows in the raw_data
    """
    split_artifact = wandb.Artifact(
        'data-splits', type='dataset',
        description='Train, validation, test dataset splits')

    # Our data split artifact will only store index references to the original dataset to save space
    split_artifact.add(wandb.Table(
        data=pd.DataFrame(train_rows, columns=['indices'])), 'train-data')

    split_artifact.add(wandb.Table(
        columns=['source'],
        data=pd.DataFrame(val_rows, columns=['indices'])), 'val-data')

    run.log_artifact(split_artifact)


def make_loaders(config, gc_data, damage_data):
    """
    Makes data loaders using a artifact containing the dataset splits (created using the make_split_artifact() function)
    The function assumes that you have created a data-splits artifact and a data-transforms artifact
    Arguments:
        config [dict] containing keys:
            batch_size (int) amount of rows (i.e. data instances) to be delivered in a single batch
    Returns:
        train_loader (PyTorch DataLoader) containing the training data
        val_loader (PyTorch DataLoader) containing the validation data
    """
    with wandb.init(project=PROJECT_NAME, job_type='package-data', config=config) as run:
        # Load transforms
        transform_dir = run.use_artifact('data-transforms:latest').download()
        transform_dict = json.load(open(os.path.join(transform_dir, 'transforms.txt')), object_pairs_hook=OrderedDict)
        composed_transforms = get_transforms(transform_dict)

        split_artifact = run.use_artifact('data-splits:latest')

        # Load splits
        # its a wandb.Table data type so we can use the get() method
        train_rows = split_artifact.get('train-data').get_column('indices', convert_to='numpy')
        val_rows = split_artifact.get('val-data').get_column('indices', convert_to='numpy')

        # Reformat data to (inputs, labels)
        train_loader = DataLoader(FractureDamageDataset(
            gc_data[train_rows], damage_data[train_rows], transform=composed_transforms),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(FractureDamageDataset(
            gc_data[val_rows], damage_data[val_rows], transform=composed_transforms),
            batch_size=config['batch_size'],
            batch_sampler=None,
            shuffle=False,
            num_workers=0)
    
    return train_loader, val_loader

# %% [markdown]
# ### Get the raw data by downloading it into a directory, load the raw data and create indices for train and val

# %%
with wandb.init(project=PROJECT_NAME, job_type="split-data", config=config) as run:

    # Define raw data splits
    raw_data = run.use_artifact('jyyresearch/FNO2D/fracture-damage-raw-data:latest', type='raw_data')

    raw_data_dir = raw_data.download()

    # read in the h5 files
    gc_data = h5py.File(os.path.join(raw_data_dir, 'gc_data'), 'r')['gc_data'][:]
    damage_data = h5py.File(os.path.join(raw_data_dir, 'damage_data'), 'r')['damage_data'][:]

    # train test split of gc_data and damage_data. Obtain the respective indices
    train_val_split = config['train_val_split']
    train_val_indices = np.split(np.random.permutation(len(gc_data)), [int(train_val_split[0]*len(gc_data))])
    
    make_split_artifact(run, train_val_indices[0], train_val_indices[1])
    

# %% [markdown]
# ### Make our dataloaders using our uploaded composed transform, and our train and val indices, and also our raw_data.

# %%
# Define an initial set of transforms that we think will be useful
with wandb.init(project=PROJECT_NAME, job_type='define-transforms', config=config) as run:
    transform_dict = OrderedDict()
    transform_dict['ToTensor'] = {
        'device': DEVICE
    }
    # Include an operational index to verify the order
    for key_idx, key in enumerate(transform_dict.keys()):
        transform_dict[key]['order'] = key_idx
    # Create an artifact for logging the transforms
    data_transform_artifact = wandb.Artifact(
        'data-transforms', type='parameters',
        description='Data preprocessing functions and parameters.',
        metadata=transform_dict) # Optional for viewing on the web app; the data is also stored in the txt file below
    # Log the transforms in JSON format
    with data_transform_artifact.new_file('transforms.txt') as f:
        f.write(json.dumps(transform_dict, indent=4))
    run.log_artifact(data_transform_artifact)

config.update(transform_dict)

train_loader, val_loader = make_loaders(config, gc_data=gc_data, damage_data=damage_data)

# %%
# config for model
step_size = 50
gamma = 0.5

modes = 12
s = modes * 4

# %%
def train(model, device, train_loader, optimizer, model_iphi, config):
    model.train()
    train_loss = 0

    myloss = LpLoss(size_average=False)

    accum_iter = config['accum_iter']

    for batch_idx, (data, damage) in enumerate(train_loader):        

        # HEAVISIDE WEIGHT FUNCTION
        # find index of values > 0.3
        # w_batch_index = np.apply_along_axis(lambda x: x > 0.3, 1, damage[:, :, 2].numpy())
        # weights_norm = np.where(w_batch_index, (0.8/np.sum(w_batch_index, axis=1))[:, np.newaxis], (0.2/np.sum(~w_batch_index, axis=1))[:, np.newaxis])
        # weights_norm = torch.from_numpy(weights_norm).float().cuda()   # to tensor

        
        
        data, damage = data.to(device), damage.to(device)
        damage_values = damage[:, :, 2]

        output = model(data, iphi=model_iphi, x_in = data[:, :, :2], x_out = damage[:, :, :2])

        if len(damage_values) == config['batch_size']:
            loss_data = myloss(output.view(config['batch_size'], -1), damage_values.view(config['batch_size'], -1))
        else:
            loss_data = myloss(output.view(len(damage_values), -1), damage_values.view(len(damage_values), -1))
        # loss = loss_data + 0.000 * loss_reg
        loss = loss_data/accum_iter
        loss.backward()

        # perform gradient accumulation
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        train_loss += loss.item()
        
    train_loss /= len(train_loader.dataset)
    return train_loss

# %%
def validate(model, device, valid_loader, model_iphi, config):

    model.eval()
    valid_loss = 0

    data_list = []
    output_list = []
    damage_list = []


    with torch.no_grad():
        myloss = LpLoss(size_average=False)
        for batch_idx, (data, damage) in enumerate(valid_loader):
            data, damage = data.to(device), damage.to(device)
            damage_values = damage[:, :, 2]

            # HEAVISIDE WEIGHT FUNCTION
            # # find index of values > 0.3
            # w_batch_index = np.apply_along_axis(lambda x: x > 0.3, 1, damage[:, :, 2].cpu().numpy())
            # weights_norm = np.where(w_batch_index, (0.8/np.sum(w_batch_index, axis=1))[:, np.newaxis], (0.2/np.sum(~w_batch_index, axis=1))[:, np.newaxis])
            # weights_norm = torch.from_numpy(weights_norm).float().cuda()

            # myloss = MSELoss_weighted(weights_tensor = weights_norm)

            output = model(data, iphi=model_iphi, x_in = data[:, :, :2], x_out = damage[:, :, :2])
            if len(damage_values) == config['batch_size']:
                loss_data = myloss(output.view(config['batch_size'], -1), damage_values.view(config['batch_size'], -1))
            else:
                loss_data = myloss(output.view(len(damage_values), -1), damage_values.view(len(damage_values), -1))
            loss = loss_data + 0
            valid_loss += loss.item()

            data_list.append(data.cpu().numpy())
            output_list.append(output.cpu().numpy())
            damage_list.append(damage.cpu().numpy())

    valid_loss /= len(valid_loader.dataset)

    return valid_loss, data_list, output_list, damage_list

# %%
# for deterministic pytorch algorithms, enable reproducibility.
os.environ['CUBLAS_WORKSPACE_CONFIG']= ":4096:8"

n_list = [32]

LOCAL_MODEL_DIR = 'Model/model_FNO2D.pt'
LOCAL_MODEL_IPHI_DIR = 'Model/model_iphi_FNO2D.pt'

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

BETA1 = config['BETA1']
BETA2 = config['BETA2']

EPOCHS = config['EPOCHS']
BATCH_SIZE = config['batch_size']

for i in range(len(n_list)):

    min_valid_loss = np.inf

    width = n_list[i]

    today = datetime.datetime.now()

    model = FNO2D.FNO2d(modes, modes, width=width, in_channels=INPUT_CHANNELS, out_channels=OUTPUT_CHANNELS, s1=s, s2=s).cuda()
    model_iphi = FNO2D.IPHI_constant(width=width).cuda()

    print(count_params(model), count_params(model_iphi))

    params = list(model.parameters()) + list(model_iphi.parameters())
    optimizer = AdamW(params, lr=config['LEARNING_RATE'], weight_decay=1e-4)
    # optimizer = dadaptation.DAdaptAdam(params, lr=1, log_every=5, betas=(BETA1, BETA2), growth_rate=1.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss_list = []
    valid_loss_list = []

    data_list = []
    output_list = []

    wandb.init(
        anonymous='allow', project=PROJECT_NAME, name= today.strftime('%Y%m%d_%H%M'),
            config={
                "epochs": EPOCHS,
                 "optimizer": 'AdamW',
                "batch_size": BATCH_SIZE * config['accum_iter'], 'lr': '1e-3',
                'step_size': step_size, 'gamma': gamma,
                'width': width,
                'modes': modes,
                'loss ': 'L2Loss',
                'activation func': 'SELU',
                'lr decay': 'steplr',
                'in_channels': INPUT_CHANNELS, 'out_channels': OUTPUT_CHANNELS,
                'architecture': 'FNO2D - IPHI - accum_gradient',
                }
        )

    for epoch in range(1, EPOCHS + 1):
        

        # get current learning rate
        curr_lr = optimizer.param_groups[0]['lr']
            
        train_loss = train(model, DEVICE, train_loader, optimizer, model_iphi, config)

        scheduler.step()    

        valid_loss, data_list, output_list, damage_list = validate(model, DEVICE, val_loader, model_iphi, config)
        print('Epoch: {:03d}, Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.7f}'.format(epoch, train_loss, valid_loss, curr_lr))
        # wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss})
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'lr': curr_lr}) 

        if valid_loss < min_valid_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_valid_loss, valid_loss))
            min_valid_loss = valid_loss
            best_epoch = epoch
            # save model with current hour as name
            today_minute = datetime.datetime.now().minute
            today_hour = datetime.datetime.now().hour
            today_day = datetime.datetime.now().day
            today_month = datetime.datetime.now().month
            torch.save(model.state_dict(), LOCAL_MODEL_DIR)
            torch.save(model_iphi.state_dict(), LOCAL_MODEL_IPHI_DIR)
            print('Saved model at epoch {}'.format(epoch))

# %%
# version control model
run = wandb.init(project=PROJECT_NAME, job_type='version-model', config=config)
trained_model_at = wandb.Artifact("FNO2D", type="model", description="trained baseline for FNO2D")
trained_model_at.add_file(LOCAL_MODEL_DIR, name='model_FNO2D.pt')
trained_model_at.add_file(LOCAL_MODEL_IPHI_DIR, name='model_iphi_FNO2D.pt')
run.log_artifact(trained_model_at)
run.finish()

# # %%
# # version control model
# run = wandb.init(project=PROJECT_NAME, job_type="inference", config=config)
# trained_model_at = run.use_artifact("FNO2D:latest", type="model")
# model_dir = trained_model_at.download()

# # load best model
# model = FNO2D.FNO2d(modes, modes, width=32, in_channels=3, out_channels=1, s1=s, s2=s).cuda()
# model_iphi = FNO2D.IPHI_constant(width=32).cuda()

# model = model.load_state_dict(torch.load(os.path.join(model_dir, 'model_FNO2D.pt')))
# model_iphi = model_iphi.load_state_dict(torch.load(os.path.join(model_dir, 'model_iphi_FNO2D.pt')))
# run.finish()

# # %%
# # fetch coordinates from wandb
# run = wandb.init(project=PROJECT_NAME, job_type="get_coordinates", config=config)
# artifact = run.use_artifact('jyyresearch/FNO2D/fracture-damage-coordinates:latest', type='coordinates')
# artifact_dir = artifact.download()

# gc_coordinates = pd.read_csv(os.path.join(artifact_dir, 'gc_out_coord'), sep=",", header=None).to_numpy()
# d_coordinates = pd.read_csv(os.path.join(artifact_dir, 'd_out_coord'), sep=",", header=None).to_numpy()

# # %% [markdown]
# # ### Perform inference

# # %%
# valid_loss, data_list, output_list, damage_list = validate(model, device, val_loader, model_iphi, config)

# # %%
# id1, id2 = 10, 6  # 75 batchs 8 per batch
# # i = id1*TEST_BATCH_SIZE + id2

# # %%
# # make sure that ids are same with different batch sizes
# # id1 is the nth batch, id2 is the nth sample in the batch




# x_n, y_n = gc_coordinates[:, 0], gc_coordinates[:, 1]
# damage_x, damage_y = d_coordinates[:, 0], d_coordinates[:, 1]

# # first subplot on left
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(x_n, y_n, c=data_list[id1][id2][:, 2], cmap='jet', s=10, vmin=0, vmax=6)
# plt.colorbar()
# plt.title('input gc field')

# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # original d
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # d
# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=output_list[id1][id2][:, 0], cmap='jet', s=10)
# plt.colorbar()
# plt.title('output d field')

# %%
# id1, id2 = 4, 6  # 75 batchs 8 per batch
# # i = id1*TEST_BATCH_SIZE + id2

# # subplots
# # gc


# x_n, y_n = coordinates.iloc[:, 0], coordinates.iloc[:, 1]

# # first subplot on left
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(x_n, y_n, c=data_list[id1][id2][:, 2], cmap='jet', s=10, vmin=0, vmax=6)
# plt.colorbar()
# plt.title('input gc field')

# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # original d
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # d
# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=output_list[id1][id2][:, 0], cmap='jet', s=10)
# plt.colorbar()
# plt.title('output d field')

# %%
# id1, id2 = 4, 15  # 75 batchs 8 per batch
# # i = id1*TEST_BATCH_SIZE + id2

# # subplots
# # gc


# x_n, y_n = coordinates.iloc[:, 0], coordinates.iloc[:, 1]

# # first subplot on left
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(x_n, y_n, c=data_list[id1][id2][:, 2], cmap='jet', s=10, vmin=0, vmax=6)
# plt.colorbar()
# plt.title('input gc field')

# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # original d
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # d
# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=output_list[id1][id2][:, 0], cmap='jet', s=10)
# plt.colorbar()
# plt.title('output d field')

# %%
# id1, id2 = 9, 15  # 75 batchs 8 per batch
# # i = id1*TEST_BATCH_SIZE + id2

# # subplots
# # gc


# x_n, y_n = coordinates.iloc[:, 0], coordinates.iloc[:, 1]

# # first subplot on left
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(x_n, y_n, c=data_list[id1][id2][:, 2], cmap='jet', s=10, vmin=0, vmax=6)
# plt.colorbar()
# plt.title('input gc field')

# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # original d
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # d
# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=output_list[id1][id2][:, 0], cmap='jet', s=10)
# plt.colorbar()
# plt.title('output d field')

# %%
# id1, id2 = 1, 10  # 75 batchs 8 per batch
# # i = id1*TEST_BATCH_SIZE + id2

# # subplots
# # gc


# x_n, y_n = coordinates.iloc[:, 0], coordinates.iloc[:, 1]

# # first subplot on left
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(x_n, y_n, c=data_list[id1][id2][:, 2], cmap='jet', s=10, vmin=0, vmax=6)
# plt.colorbar()
# plt.title('input gc field')

# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # original d
# plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 1)
# plt.scatter(damage_x, damage_y, c=damage_list[id1][id2][:,2], cmap='jet', s=10)
# plt.colorbar()
# plt.title('Original d field')

# # d
# # plt.figure(figsize=(18, 7.5))
# plt.subplot(1, 2, 2)
# plt.scatter(damage_x, damage_y, c=output_list[id1][id2][:, 0], cmap='jet', s=10)
# plt.colorbar()
# plt.title('output d field')


