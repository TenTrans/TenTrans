import torch
import torch.utils.data as data
from src.utils.utility import batch_data
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseTextDataSet(data.Dataset):
    def __init__(self, items, data_config, src_vocab):
        self.items = items
        self.data_flow = {}
        self.lengths = []

    def check_data(self):
        logger.info("checking data....")

        for i, item in enumerate(self.items):
            item.apply(self.data_flow)
            if i == 1: logger.info(item.process_content)
            self.lengths.append(len(item.process_content[0]))

        self.lengths = np.asarray(self.lengths)

    def __getitem__(self, index):
        return self.items[index].apply(self.data_flow)

    def __len__(self):
        return len(self.items)

    def _build_process_flow(self):
        raise NotImplementedError

    def collate_fn(self, data):
        raise NotImplementedError
