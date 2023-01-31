"""
Evaluation helper functions for pretrained models
"""

import numpy as np
import torch
from utils.logging import summarize_acc_from_predictions

def evaluate_waterbirds_predictions(predictions, dataloader):
    targets  = dataloader.dataset.targets_all['target']
    spurious = dataloader.dataset.targets_all['spurious']
    
    try:
        predictions = predictions.numpy()
    except:
        pass
    correct_by_group = [[0, 0], [0, 0]]
    total_by_group   = [[0, 0], [0, 0]]
    accs_by_group    = [[0, 0], [0, 0]]
    correct = predictions == targets
    for t in [0, 1]:
        for s in [0 ,1]:
            ix = np.where(np.logical_and(targets == t,
                                         spurious == s))[0]
            correct_by_group[t][s] += np.sum(correct[ix])
            total_by_group[t][s] += len(ix)
            accs_by_group[t][s] = np.sum(correct[ix]) / len(ix)
    
    # Average accuracy
    avg_acc = (
        correct_by_group[0][0] +
        correct_by_group[0][1] +
        correct_by_group[1][0] +
        correct_by_group[1][1]
    )
    avg_acc = avg_acc * 100 / np.sum(np.array(total_by_group))
    
    # Adjust average accuracy
    adj_avg_acc = (
        accs_by_group[0][0] * 3498 +
        accs_by_group[0][1] * 184 +
        accs_by_group[1][0] * 56 +
        accs_by_group[1][1] * 1057
    )
    adj_avg_acc = adj_avg_acc * 100 / (3498 + 184 + 56 + 1057)
    
    accs_by_group = np.array(accs_by_group).flatten() * 100
    
    worst_acc = np.min(accs_by_group)
    
    return worst_acc, adj_avg_acc, avg_acc, accs_by_group


def evaluate_dataset_prediction(predictions, dataloader, 
                                args, verbose=True):
    if args.dataset == 'celebA':
        try:
            predictions = predictions.cpu().numpy()
        except:
            pass
        avg_acc, min_acc = summarize_acc_from_predictions(
            predictions, dataloader, args, stdout=verbose
        )
    elif args.dataset == 'waterbirds':
        accs = evaluate_waterbirds_predictions(predictions, 
                                               dataloader)
        worst_acc, adj_avg_acc, avg_acc_, accs_by_group = accs
        avg_acc = adj_avg_acc
        min_acc = worst_acc
        if verbose:
            for ix, acc in enumerate(accs_by_group):
                print(f'Group {ix} acc: {acc:.2f}%')
            print(f'Worst-group acc: {worst_acc:.2f}%')
            print(f'Average acc:     {avg_acc_:.2f}%')
            print(f'Adj Average acc: {adj_avg_acc:.2f}%')
    else:
        print('Not Support')
        avg_acc, min_acc = None, None
    return avg_acc, min_acc


    
