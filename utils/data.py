from typing import List, Tuple
import torch
import numpy as np
from itertools import chain


def custom_collate(l: List[Tuple[torch.FloatTensor, torch.LongTensor, np.float32]]) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
    """
    Custom collate function for pytorch dataloaders. Each data point is a collection of atoms and the model evaluates
    energetic contributions in an atomwise manner. This function, on top of return a batch's x and y, will also return
    a batch_index tensor that represents which atoms belong to which sample

    TODO: pretty naive implementation; this could probably be faster

    :param l: list of tuple of (x, y)
    :return:
    """
    # x = []
    # species_idx = []
    # y = []
    # batch_idx = []
    # for idx, i in enumerate(l):
    #     x.append(i[0])
    #     species_idx.append(i[1])
    #     y.append(i[2])
    #     batch_idx.extend([idx]*len(i[0]))
    x, species_idx, y = zip(*l)  # kinda like a transpose operation
    batch_idx = list(chain.from_iterable(  # chain.from_iterable basically flattens a list of lists
        [[i]*len(l[i][0]) for i in range(len(l))]
    ))
    return torch.cat(x, dim=0), torch.cat(species_idx, dim=0), torch.FloatTensor(y), torch.LongTensor(batch_idx)
