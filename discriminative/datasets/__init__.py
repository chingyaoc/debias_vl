"""
Datasets and data utils

Functions:
- initialize_data()
- train_val_split()
- get_resampled_set()
- imshow()
- plot_data_batch()
"""
import copy
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image


def initialize_data(args):
    """
    Set dataset-specific default arguments
    """
    args.resample_iid = False
    if 'cifar10e' in args.dataset:
        dataset = 'cifar10e'
    else:
        dataset = args.dataset.split('_')[0]
    dataset_module = importlib.import_module(f'datasets.{dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')
    visualize_dataset = getattr(dataset_module, 'visualize_dataset')
    
    args.results_dict = {'epoch': [],
                         'dataset_ix': [],  # legacy
                         'train_loss': [],
                         'train_avg_acc': [],
                         'train_robust_acc': [],
                         'val_loss': [],
                         'val_avg_acc': [],
                         'val_robust_acc': [],
                         'test_loss': [],
                         'test_avg_acc': [],
                         'test_robust_acc': [],
                         'best_loss_epoch': [],
                         'best_acc_epoch': [],
                         'best_robust_acc_epoch': []}
        
    if args.dataset == 'waterbirds':
        # Update this to right path
        args.root_dir = './datasets/data/Waterbirds/' 
        args.val_split = 0.2
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.augment_data = False
        args.train_classes = ['landbird', 'waterbird']
        ## Image
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.text_descriptions = ['a landbird', 'a waterbird']
        args.wilds_dataset = False
        
    elif 'celebA' in args.dataset:
        args.root_dir = './datasets/data/CelebA/'
        # IMPORTANT - dataloader assumes that we have directory structure
        # in ./datasets/data/CelebA/ :
        # |-- list_attr_celeba.csv
        # |-- list_eval_partition.csv
        # |-- img_align_celeba/
        #     |-- image1.png
        #     |-- ...
        #     |-- imageN.png
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = './images/celebA/'
        args.train_classes = ['nonblond', 'blond']
        args.val_split = 0.2
        args.text_descriptions = ['a celebrity with dark hair',
                                  'a celebrity with blond hair']
        args.wilds_dataset = False
        
    else:
        raise NotImplementedError
    
    args.task = args.dataset 
    try:
        args.num_classes = len(args.train_classes)
    except:
        pass
    return load_dataloaders, visualize_dataset




def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def imshow(img, mean=0.5, std=0.5):
    """
    Visualize data batches
    """
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_data_batch(dataset, mean=0.0, std=1.0, nrow=8, title=None,
                    args=None, save=False, save_id=None, ftype='png'):
    """
    Visualize data batches
    """
    try:
        img = make_grid(dataset, nrow=nrow)
    except Exception as e:
        raise e
        print(f'Nothing to plot!')
        return
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if save:
        try:
            fpath = join(args.image_path,
                         f'{save_id}-{args.experiment_name}.{ftype}')
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        except Exception as e:
            fpath = f'{save_id}-{args.experiment_name}.{ftype}'
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
    if args.display_image:
        plt.show()
    plt.close()


