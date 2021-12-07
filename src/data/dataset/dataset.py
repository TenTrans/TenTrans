import logging
import numpy as np
import torch.utils.data as data

logger = logging.getLogger(__name__)


class BaseTextDataSet(data.Dataset):
    def __init__(self, items, data_config, src_vocab):
        self.items = items
        self.data_flow = {}
        self.lengths = []

    def check_data(self):
        logger.info("checking data....")
        index = list(self.items[0].content.keys()).index("seq1")
        for i, item in enumerate(self.items):
            item.apply(self.data_flow)
            if i == 1:
                logger.info(item.process_content)
            if i % 100000 == 0:
                logger.info(i)
            self.lengths.append(len(item.process_content[index]))
        self.lengths = np.asarray(self.lengths)

    def __getitem__(self, index):
        return self.items[index].apply(self.data_flow)

    def __len__(self):
        return len(self.items)

    def _build_process_flow(self):
        raise NotImplementedError

    def collate_fn(self, data):
        raise NotImplementedError
