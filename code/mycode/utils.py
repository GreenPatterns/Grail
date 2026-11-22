import torch
import scipy
import warnings
import numpy as np
import networkx as nx
from texttable import Texttable
warnings.filterwarnings("ignore", module="sklearn.metrics")
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef

import numpy as np
import scipy.sparse
import networkx as nx

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def read_graph(args):
    edges = {}
    ecount = 0
    ncount = []
    edg = []
    lab = []
    with open(args.edge_path) as dataset:
        for edge in dataset:
            ecount += 1
            ncount.append(edge.split()[0])
            ncount.append(edge.split()[1])
            edg.append(list(map(float, edge.split()[0:2])))
            lab.append(list(map(float, edge.split()[2:])))
    edges["labels"] = np.array(lab)
    edges["edges"] = np.array(edg) 
    edges["ecount"] = ecount
    edges["ncount"] = len(set(ncount)) # Node Count
    return edges

# RFP
# def setup_features(graph_data, seed, dim=16):
#     rng = np.random.default_rng(seed)
#     num_nodes = graph_data['ncount']
#     G = nx.Graph()
#     G.add_nodes_from(range(num_nodes))
#     G.add_edges_from(graph_data['edges'])
#     A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
#     A = A + scipy.sparse.eye(num_nodes) 
#     deg = np.array(A.sum(axis=1)).flatten()
#     deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
#     D_inv_sqrt = scipy.sparse.diags(deg_inv_sqrt)
#     A_hat = D_inv_sqrt @ A @ D_inv_sqrt  
#     X = rng.normal(0, 1, (num_nodes, dim))
#     num_steps = 3
#     trajectory = [X]
#     for _ in range(num_steps):
#         X = A_hat @ X 
#         trajectory.append(X)
#     embeddings = np.mean(trajectory, axis=0) 
#     mu = embeddings.mean(axis=0, keepdims=True)
#     sigma = embeddings.std(axis=0, keepdims=True) 
#     embeddings = (embeddings - mu) / sigma
#     return embeddings.astype(np.float32)


import matplotlib.pyplot as plt
def plot_dict_line(data_dict, filename="plot.png", title="Line Plot", xlabel="Keys", ylabel="Values"):
    """
    Plots the values of a dictionary as a line plot and saves the figure.

    Converts any torch tensors on GPU to CPU before plotting.
    """
    if not data_dict:
        raise ValueError("The dictionary is empty.")

    # Sort dictionary by keys
    keys = sorted(data_dict.keys())
    values = [data_dict[k] for k in keys]

    # Convert torch tensors to CPU numpy arrays if needed
    for i, v in enumerate(values):
        if isinstance(v, torch.Tensor):
            values[i] = v.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(keys, values, marker='o', linestyle='-', linewidth=2, markersize=6, color='dodgerblue')
    plt.fill_between(keys, values, color='dodgerblue', alpha=0.2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def setup_features(graph_data, seed, dim=16, alpha=5.0):
    rng = np.random.default_rng(seed)
    num_nodes = graph_data['ncount']
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(graph_data['edges'])
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    A = A + scipy.sparse.eye(num_nodes)
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = scipy.sparse.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    X = rng.normal(0, 1, (num_nodes, dim))
    trajectory = [X]
    steps_per_node = np.maximum(1, np.floor(alpha / (deg + 1)).astype(int))
    max_steps = int(steps_per_node.max())
    X_cur = X.copy()
    active = np.ones(num_nodes, dtype=bool)
    for step in range(max_steps):
        X_next = X_cur.copy()
        X_next[active] = A_hat[active][:, :] @ X_cur
        trajectory.append(X_next)
        steps_per_node[active] -= 1
        active &= (steps_per_node > 0)
        if not active.any():
            break
        X_cur = X_next
    embeddings = np.mean(trajectory, axis=0)
    mu = embeddings.mean(axis=0, keepdims=True)
    sigma = embeddings.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    embeddings = (embeddings - mu) / sigma
    return embeddings.astype(np.float32)


def calculate_auc(predictions, test_labels):
    if torch.isnan(predictions).any() :
        raise ValueError("predictions contains NaN values")  
    label_vector = [i for line in test_labels for i in range(len(line)) if line[i] == 1]
    prediction_vector = torch.argmax(predictions, dim=1)
    prediction_vector =  prediction_vector.cpu()
    prediction_pr = predictions[:,1]
    prediction_pr = prediction_pr.cpu().detach().numpy().astype(np.float64)
    f1_micro = f1_score(label_vector, prediction_vector, average="micro")
    f1_macro = f1_score(label_vector, prediction_vector, average="macro")
    acc_balanced = balanced_accuracy_score(label_vector,prediction_vector)
    mcc = matthews_corrcoef(label_vector, prediction_vector)
    auc = roc_auc_score(label_vector, prediction_pr)
    precision = average_precision_score(label_vector, prediction_pr)  

    return mcc, auc, acc_balanced, precision, f1_micro, f1_macro

def print_config(args):
    print("\n" + "="*55)
    print(f"Dataset: {args.dataset}")

    print(
        f"Task: {'Single' if args.single_prediction else 'Multiple'} | "
        f"Task3: {'On' if args.enable_task3 else 'Off'}"
    )
    print(
        f"Attack: {'On (' + args.attack_type + ')' if args.attack else 'Off'}"
    )

    print("="*55 + "\n")

def best_printer(log):
    t = Texttable()
    t.set_precision(4)
    t.add_rows([per for per in log])
    print(t.draw())

def printAvg(file):
    import pandas as pd
    df = pd.read_csv(file, skipinitialspace=True)
    selected_columns = df.loc[:, "MCC": "Run_Time"]
    means = selected_columns.mean()
    maxes = selected_columns.max()
    print("\nMean and max of all the entries in the save file:")
    print(", ".join(f"avg_{c}:{mn:.4f}" for c, mn in means.items()))
    print(", ".join(f"max_{c}:{mn:.4f}" for c, mn in maxes.items()))

    return means, maxes


def getBestGPU():
    import pynvml
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return None
        gpu_data = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_data.append({
                'index': i,
                'free_mem_bytes': mem_info.free,
                'gpu_util': util.gpu
            })
        idle_gpus = [gpu for gpu in gpu_data if gpu['gpu_util'] < 10]
        if idle_gpus:
            gpus_to_consider = idle_gpus
        else:
            gpus_to_consider = gpu_data
        best_gpu = max(gpus_to_consider, key=lambda x: x['free_mem_bytes'])
        return best_gpu['index']
    except pynvml.NVMLError:
        return None
    finally:
        pynvml.nvmlShutdown()


def save_message_to_file(filename, message):
    if isinstance(message, str):
        msg = message
    else:
        msg = ', '.join([f'{x:.4f}' for x in message])
    with open(filename, 'a') as f:
        f.write(f"{msg}\n")


def build_training_tensors(processed_snapshots, train_keys, device):
    """
    Concatenates across ALL training snapshots:
        train_edges       [2, E_total]       original edges       → main loss
        target            [E_total]          original labels      → main loss
        new_edges         [2, E_new_total]   new edges only       → aux loss
        new_edge_labels   [E_new_total, 2]   soft labels          → aux loss
        new_edge_weights  [E_new_total]      coherence weights    → aux loss
    """
    train_edges_list      = []
    target_list           = []
    new_edges_list        = []
    new_edge_labels_list  = []
    new_edge_weights_list = []

    for k in train_keys:
        snap     = processed_snapshots[k]
        new_mask = ~snap["edge_mask"]   # True = new estimated edge

        train_edges_list.append(snap["edge_label_index"])
        target_list.append(snap["edge_label_original"].long())

        if new_mask.sum() > 0:
            new_edges_list.append(snap["edge_index"][:, new_mask])
            new_edge_labels_list.append(snap["edge_label"][new_mask])
            new_edge_weights_list.append(snap["edge_coh"][new_mask])

    train_edges      = torch.cat(train_edges_list, dim=1).to(device)
    target           = torch.cat(target_list,      dim=0).to(device).long()

    new_edges        = torch.cat(new_edges_list,        dim=1).to(device) if new_edges_list        else None
    new_edge_labels  = torch.cat(new_edge_labels_list,  dim=0).to(device) if new_edge_labels_list  else None
    new_edge_weights = torch.cat(new_edge_weights_list, dim=0).to(device) if new_edge_weights_list else None

    print(f"train_edges      : {tuple(train_edges.shape)}")
    print(f"target           : {tuple(target.shape)}")
    print(f"new_edges        : {tuple(new_edges.shape)        if new_edges        is not None else None}")
    print(f"new_edge_labels  : {tuple(new_edge_labels.shape)  if new_edge_labels  is not None else None}")
    print(f"new_edge_weights : {tuple(new_edge_weights.shape) if new_edge_weights is not None else None}")

    return train_edges, target, new_edges, new_edge_labels, new_edge_weights
