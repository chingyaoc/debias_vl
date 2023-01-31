"""
CelebA Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
from utils.visualize import plot_data_batch
from copy import deepcopy


class CelebA(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406), 
                            'std': (0.229, 0.224, 0.225)}

    def __init__(self, root_dir, 
                 target_name='Blond_Hair', confounder_names=['Male'],
                 split='train', augment_data=False, model_type=None,
                 train_transform=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names 
        # Only support 1 confounder for now as in official benchmark
        confounder_names = self.confounder_names[0]  
        self.model_type = model_type
        if '_pt' in model_type:
            self.model_type = model_type[:-3]
        self.augment_data = augment_data
        self.split = split

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        
        self.data_dir = self.root_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'list_attr_celeba.csv'))#, delim_whitespace=True)
        self.split_df = pd.read_csv(os.path.join(self.data_dir, 'list_eval_partition.csv'))#, delim_whitespace=True)
        # Filter for data split ('train', 'val', 'test')
        self.metadata_df['partition'] = self.split_df['partition']
        self.metadata_df = self.metadata_df[
            self.split_df['partition'] == self.split_dict[self.split]]

        # Get the y values
        self.y_array = self.metadata_df[self.target_name].values
        self.confounder_array = self.metadata_df[confounder_names].values
        self.y_array[self.y_array == -1] = 0
        self.confounder_array[self.confounder_array == -1] = 0
        self.n_classes = len(np.unique(self.y_array))
        self.n_confounders = len(confounder_names)
        
        # Get sub_targets / group_idx
        self.metadata_df['sub_target'] = (
            self.metadata_df[self.target_name].astype(str) + '_' +
            self.metadata_df[confounder_names].astype(str))
        
        # Get subclass map
        attributes = [self.target_name, confounder_names]
        self.df_groups = (self.metadata_df[
            attributes].groupby(attributes).size().reset_index())
        self.df_groups['group_id'] = (
            self.df_groups[self.target_name].astype(str) + '_' +
            self.df_groups[confounder_names].astype(str))
        self.subclass_map = self.df_groups[
            'group_id'].reset_index().set_index('group_id').to_dict()['index']
        self.group_array = self.metadata_df['sub_target'].map(self.subclass_map).values
        groups, group_counts = np.unique(self.group_array, return_counts=True)
        self.n_groups = len(groups)

        # Extract filenames and splits
        self.filename_array = self.metadata_df['image_id'].values
        self.split_array = self.metadata_df['partition'].values

        self.targets = torch.tensor(self.y_array)
        self.targets_all = {'target': np.array(self.y_array),
                            'group_idx': np.array(self.group_array),
                            'spurious': np.array(self.confounder_array),
                            'sub_target': np.array(list(zip(self.y_array, self.confounder_array)))}
        self.group_labels = [self.group_str(i) for i in range(self.n_groups)]
        self.features_mat = None
        
        # Image transforms
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = get_transform_celeba(self.model_type, 
                                                        train=True)
            
        if train_transform is not None:
            self.eval_transform = train_transform
        else:
            self.eval_transform = get_transform_celeba(self.model_type, 
                                                        train=True)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.targets[idx] 
        img_filename = os.path.join(
            self.data_dir,
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename)
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)
        x = img

        return (x, y, idx)

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


def get_transform_celeba(model_type, train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=CelebA._normalization_stats['mean'], std=CelebA._normalization_stats['std']),
    ])
    return transform


def load_celeba(args, train_shuffle=True, transform=None):
    """
    Default dataloader setup for CelebA

    Args:
    - args (argparse): Experiment arguments
    - train_shuffle (bool): Whether to shuffle training data
    Returns:
    - (train_loader, val_loader, test_loader): Tuple of dataloaders for each split
    """
    train_set = CelebA(args.root_dir, split='train', model_type=args.arch,
                       train_transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.bs_trn,
                              shuffle=train_shuffle, num_workers=args.num_workers)

    val_set = CelebA(args.root_dir, split='val', model_type=args.arch,
                     train_transform=transform)
    val_loader = DataLoader(val_set, batch_size=args.bs_val,
                            shuffle=False, num_workers=args.num_workers)

    test_set = CelebA(args.root_dir, split='test', model_type=args.arch,
                      train_transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.bs_val,
                             shuffle=False, num_workers=args.num_workers)
    args.num_classes = 2

    return (train_loader, val_loader, test_loader)


def visualize_celebA(dataloader, num_datapoints, title, args, save,
                     save_id, ftype='png', target_type='group_idx'):
    # Filter for selected datapoints (in case we use SubsetRandomSampler)
    try:
        subset_indices = dataloader.sampler.indices
        targets = dataloader.dataset.targets_all[target_type][subset_indices]
        subset = True
    except AttributeError:
        targets = dataloader.dataset.targets_all[target_type]
        subset = False
    all_data_indices = []
    for class_ in np.unique(targets):
        class_indices = np.where(targets == class_)[0]
        all_data_indices.extend(class_indices[:num_datapoints])
    
    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in all_data_indices],
                    mean=np.mean([0.485, 0.456, 0.406]),
                    std=np.mean([0.229, 0.224, 0.225]), nrow=8, title=title,
                    args=args, save=save, save_id=save_id, ftype=ftype)

   
def load_dataloaders(args, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_celeba(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                            save_id, ftype='png', target_type='target'):
    return visualize_celebA(dataloader, num_datapoints, title, 
                            args, save, save_id, ftype, target_type)
