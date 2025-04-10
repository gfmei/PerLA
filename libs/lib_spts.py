import torch
import numpy as np

def num_to_natural_numpy(group_ids, void_number=-1):
    """
    Convert group ids to natural numbers using NumPy, handling void numbers as specified.
    Equivalent to the provided PyTorch implementation.
    """
    group_ids_array = np.array(group_ids, dtype=int)

    if void_number == -1:
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids_array == -1):
            return group_ids_array
        array_ids = group_ids_array.copy()

        unique_values = np.unique(array_ids[array_ids != -1])
        mapping = np.full(np.max(unique_values) + 2, -1, dtype=int)
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array_ids = mapping[array_ids + 1]

    elif void_number == 0:
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if np.all(group_ids_array == 0):
            return group_ids_array
        array_ids = group_ids_array.copy()

        unique_values = np.unique(array_ids[array_ids != 0])
        mapping = np.zeros(np.max(unique_values) + 2, dtype=int)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array_ids = mapping[array_ids]
    else:
        raise ValueError("void_number must be -1 or 0")

    return array_ids


def num_to_natural_torch(group_ids, void_number=-1):
    """
    Convert group ids to natural numbers, handling void numbers as specified.
    code credit: SAM3D
    """
    group_ids_tensor = group_ids.long()
    device = group_ids_tensor.device

    if void_number == -1:
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if torch.all(group_ids_tensor == -1):
            return group_ids_tensor
        array_ids = group_ids_tensor.clone()

        unique_values = torch.unique(array_ids[array_ids != -1])
        mapping = torch.full((torch.max(unique_values) + 2,), -1, dtype=torch.long, device=device)
        # map ith (start from 0) group_id to i
        mapping[unique_values + 1] = torch.arange(len(unique_values), dtype=torch.long, device=device)
        array_ids = mapping[array_ids + 1]

    elif void_number == 0:
        # [0,3,4,0,6] -> [0,1,2,0,3]
        if torch.all(group_ids_tensor == 0):
            return group_ids_tensor
        array_ids = group_ids_tensor.clone()

        unique_values = torch.unique(array_ids[array_ids != 0])
        mapping = torch.full((torch.max(unique_values) + 2,), 0, dtype=torch.long, device=device)
        mapping[unique_values] = torch.arange(len(unique_values), dtype=torch.long, device=device) + 1
        array_ids = mapping[array_ids]
    else:
        raise Exception("void_number must be -1 or 0")

    return array_ids