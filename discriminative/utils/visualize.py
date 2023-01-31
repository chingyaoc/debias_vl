"""
Functions for visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def visualize_embeddings(embeddings, dataloader,
                         color_by='target', cmap='hsv',
                         size=0.1, alpha=0.5, title=None,
                         figsize=(12, 9), display=False,
                         save_path=''):
    labels = dataloader.dataset.targets_all[color_by]
    colors = np.array(labels).astype(int)
    num_colors = len(np.unique(colors))
    plt.figure(figsize=figsize)
    plt.scatter(embeddings[:, 0], embeddings[:, 1],
                c=colors, s=size, alpha=alpha,
                cmap=plt.cm.get_cmap(cmap, num_colors))
    cbar = plt.colorbar(ticks=np.unique(colors))
    cbar = cbar.set_alpha(1) 
    if title is not None:
        plt.title(title)
    if display:
        plt.show()
    if save_path != '':
        plt.savefig(fname=save_path, dpi=300, bbox_inches="tight")
        
        
def plot_data_batch(dataset, mean=0.0, std=1.0, nrow=8, title=None,
                    args=None, save=False, save_id=None, ftype='png',
                    figsize=None):
    """
    Visualize data batches
    """
    try:
        img = make_grid(dataset, nrow=nrow)
    except:
        print(f'Nothing to plot!')
        return
    img = img * std + mean  # unnormalize
    npimg = img.numpy()
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
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
    
    
def plot_data_batch_(dataset, nrow=8, title=None,
                     args=None, save=False, save_id=None, ftype='png',
                     figsize=None):
    """
    Visualize data batches
    """
    try:
        img = make_grid(dataset, nrow=nrow)
    except:
        print(f'Nothing to plot!')
        return
    img = img * args.image_std + args.image_mean  # unnormalize
    npimg = img.numpy()
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
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
        