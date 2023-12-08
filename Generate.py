import math
import numpy as np
import random
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size

def generate_triplets(y, generate):
    stochastic = np.zeros_like(y)
    idx = random.sample(range(y.shape[0]), len(y))
    stochastic[idx] = y[idx]
    anchor = []
    positive = []
    negative = []
    num = 1

    while num > 0:
        idx1 = random.sample(idx, 2)
        if stochastic[idx1[0]] == stochastic[idx1[1]]:
            idx2 = random.sample(idx, 1)
            if stochastic[idx1[0]] != stochastic[idx2]:
                anchor.append(idx1[0])
                positive.append(idx1[1])
                negative.append(idx2)
        if len(anchor) == generate:
            break
    return np.array(anchor), np.array(positive), np.array(negative).flatten()

def cell_select(X,y,p,ratio):
    n=len(y)
    n_sub=int(n*ratio)
    mask=np.zeros(n,dtype=np.bool)
    indices=random.sample(range(n),n_sub)
    mask[indices]=True
    X_sub=X[mask,:]
    y_sub = y[mask]
    p_sub = p[mask,:]
    return X_sub,y_sub,p_sub

def adjust_labels(labels):
    unique_labels = np.unique(labels)  # Find all the different tags and sort them
    unique_labels_sorted = np.sort(unique_labels)
    new_labels = np.zeros_like(labels, dtype=np.int64)  # Create a new label tensor with data type long
    for i in range(len(labels)):  # Adjust label range
        label = labels[i]
        new_label = np.where(unique_labels_sorted == label)[0][0]
        new_labels[i] = new_label
    return new_labels