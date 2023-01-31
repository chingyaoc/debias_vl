import os
import numpy as np
import torch

### Logging
def summarize_acc(correct_by_groups, total_by_groups, 
                  stdout=True, return_groups=False):
    all_correct = 0
    all_total = 0
    min_acc = 101.
    min_correct_total = [None, None]
    groups_accs = np.zeros([len(correct_by_groups), 
                            len(correct_by_groups[-1])])
    if stdout:
        print('Accuracies by groups:')
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            acc = a_group / total_by_groups[yix][aix] * 100
            groups_accs[yix][aix] = acc
            # Don't report min accuracy if there's no group datapoints
            if acc < min_acc and total_by_groups[yix][aix] > 0:
                min_acc = acc
                min_correct_total[0] = a_group
                min_correct_total[1] = total_by_groups[yix][aix]
            if stdout:
                print(
                    f'{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}')
            all_correct += a_group
            all_total += total_by_groups[yix][aix]
    if stdout:
        average_str = f'Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
        robust_str = f'Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
        print('-' * len(average_str))
        print(average_str)
        print(robust_str)
        print('-' * len(average_str))
        
    avg_acc = all_correct / all_total * 100
        
    if return_groups:
        return avg_acc, min_acc, groups_accs
    return avg_acc, min_acc 


def summarize_acc_from_predictions(predictions, dataloader,
                                   args, stdout=True, 
                                   return_groups=False):
    targets_t = dataloader.dataset.targets_all['target']
    targets_s = dataloader.dataset.targets_all['spurious']
    
    correct_by_groups = np.zeros([args.num_classes,
                                  args.num_classes])
    total_by_groups = np.zeros(correct_by_groups.shape)
    
    all_correct = (predictions == targets_t)
    for ix, s in enumerate(targets_s):
        y = targets_t[ix]
        correct_by_groups[int(y)][int(s)] += all_correct[ix]
        total_by_groups[int(y)][int(s)] += 1
    return summarize_acc(correct_by_groups, total_by_groups,
                         stdout=stdout, return_groups=return_groups)


def log_metrics(train_metrics, val_metrics, test_metrics, epoch,
                dataset_ix=0, args=None):
    assert args is not None
    if args.wilds_dataset:
        pass
    
    train_loss, correct, total, correct_by_groups, total_by_groups = train_metrics
    train_avg_acc, train_min_acc = summarize_acc(correct_by_groups,
                                                 total_by_groups,
                                                 stdout=args.verbose)
#     print(f'Train epoch {epoch} | loss: {train_loss:<3.2f} | avg acc: {avg_acc:<.2f}% | robust acc: {min_acc:<.2f}%')
    args.results_dict['epoch'].append(epoch)
    args.results_dict['dataset_ix'].append(dataset_ix)
    args.results_dict['train_loss'].append(train_loss)
    args.results_dict['train_avg_acc'].append(train_avg_acc)
    args.results_dict['train_robust_acc'].append(train_min_acc)

    val_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
    val_avg_acc, val_min_acc = summarize_acc(correct_by_groups,
                                             total_by_groups,
                                             stdout=args.verbose)
#     print(f'Val   epoch {epoch} | loss: {val_loss:<3.2f} | avg acc: {avg_acc:<.2f}% | robust acc: {min_acc:<.2f}%')
    args.results_dict['val_loss'].append(val_loss)
    args.results_dict['val_avg_acc'].append(val_avg_acc)
    args.results_dict['val_robust_acc'].append(val_min_acc)

    loss, correct, total, correct_by_groups, total_by_groups = test_metrics
    avg_acc, min_acc = summarize_acc(correct_by_groups,
                                     total_by_groups,
                                     stdout=args.verbose)
#     print(f'Test  epoch {epoch} | loss: {loss:<3.2f} | avg acc: {avg_acc:<.2f}% | robust acc: {min_acc:<.2f}%')
    args.results_dict['test_loss'].append(loss)
    args.results_dict['test_avg_acc'].append(avg_acc)
    args.results_dict['test_robust_acc'].append(min_acc)
    
    train_metrics = (train_loss, train_avg_acc, train_min_acc)
    val_metrics = (val_loss, val_avg_acc, val_min_acc)
    return train_metrics, val_metrics


    
def log_data(dataset, header, indices=None):
    print(header)
    dataset_groups = dataset.targets_all['group_idx']
    if indices is not None:
        dataset_groups = dataset_groups[indices]
    groups = np.unique(dataset_groups)
    
    try:
        max_target_name_len = np.max([len(x) for x in dataset.class_names])
    except Exception as e:
        print(e)
        max_target_name_len = -1
    
    for group_idx in groups:
        counts = np.where(dataset_groups == group_idx)[0].shape[0]
        try:  # Arguably more pretty stdout
            group_name = dataset.group_labels[group_idx]
            group_name = group_name.split(',')
            group_name[0] += (' ' * int(
                np.max((0, max_target_name_len - len(group_name[0])))
            ))
            group_name = ','.join(group_name)
            print(f'- {group_name} : n = {counts}')
        except Exception as e:
            print(e)
            print(f'- {group_idx} : n = {counts}')
