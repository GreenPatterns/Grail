import pandas as pd
import datetime
import numpy as np

def get_snapshot_index(time_slots, data_path, homogeneous_edges=True):
    if homogeneous_edges:
        return get_homogeneous_snapshot_index(time_slots, data_path)
    else:
        return get_heterogeneous_snapshot_index(time_slots, data_path)


def get_homogeneous_snapshot_index(time_slots, data_path):
    # print(f"Getting homogeneous snapshot indices for {time_slots} time slots from {data_path}")
    df = pd.read_csv(data_path)
    df.columns = ['source', 'target', 'rating', 'time']
    df = df.sort_values('time', kind='mergesort').reset_index(drop=True)

    total_edges = len(df)
    
    if total_edges < time_slots:
        return list(range(total_edges - 1))
    
    edges_per_snapshot = total_edges // time_slots
    remainder = total_edges % time_slots
    
    index_list = []
    current_end_index = -1
    for i in range(time_slots):
        edges_in_this_snapshot = edges_per_snapshot + (1 if i < remainder else 0)
        current_end_index += edges_in_this_snapshot
        index_list.append(current_end_index)
    
    index_list[-1] = total_edges - 1
    return index_list


def get_heterogeneous_snapshot_index(time_slots, data_path):
    """Get snapshot indices with unequal number of edges per snapshot based on time."""
    # print(f"Getting heterogeneous snapshot indices for {time_slots} time slots from {data_path}")   
    df = pd.read_csv(data_path)
    df.columns = ['source', 'target', 'rating', 'time']
    time_list = np.sort(df['time'].to_numpy())
 
    ts_begin = time_list[0]
    ts_finish = time_list[-1]

    span = ts_finish - ts_begin
    split_list = []
    
    for i in range(1, time_slots):
        split_list.append(ts_begin + (span * i) / time_slots)
    split_list.append(ts_finish)

    index_list = []
    for boundary in split_list:
        if boundary == split_list[-1]:
            index_list.append(len(time_list) - 1)
            break

        split_index = int(np.searchsorted(time_list, boundary, side='right') - 1)
        split_index = max(split_index, 0)
        index_list.append(split_index)

    return index_list
