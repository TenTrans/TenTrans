from torch.utils.data import Sampler
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
import numpy as np
import math
import torch.distributed as dist
import torch



class BucketBatchSampler(Sampler):

    def __init__(self, dataset, shuffle: bool = True, batch_size: int = 32, max_tokens: int = -1, group_by_size: bool = False):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.dataset = dataset
        self.group_by_size = group_by_size

    def __iter__(self):
        
        if self.shuffle:
            rng = np.random.RandomState(None)
            indices = rng.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        if self.group_by_size:
            indices = indices[np.argsort(self.dataset.lengths[indices], kind='mergesort')]

        if self.max_tokens == -1:
            batches = np.array_split(indices, math.ceil(len(indices)/self.batch_size))
        else:
            batch_ids = np.cumsum(self.dataset.lengths[indices]) // self.max_tokens
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])
        
        if self.shuffle:
            rng.shuffle(batches)

        for sentence_ids in iter(batches):
            yield sentence_ids
    