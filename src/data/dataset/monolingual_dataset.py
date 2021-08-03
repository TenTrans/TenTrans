import torch
import torch.utils.data as data
from src.utils.utility import batch_data
from .dataset import BaseTextDataSet
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StreamBinaryDataset(data.Dataset):
    def __init__(self, items, data_config, src_vocab):

        assert items['word2id'] == src_vocab.stoi
        self.pos = items['positions']
        self.sents = items['sents']
        self.lang = items['lang']
        self.lang_id = src_vocab.index(f"[[{self.lang}]]")

        self.pad_index = src_vocab.pad_index
        self.eos_index = src_vocab.eos_index
        self.bptt = data_config.get('bptt', 256)
        self.batch_size = data_config['batch_size']

        # This practice follows the XLM
        # (https://github.com/facebookresearch/XLM/blob/0e1bde1ef24847b00dba4061ed59f8fb6578e486/xlm/data/dataset.py#L17).
        n_tokens = len(self.sents)
        bs = self.batch_size
        n_batches = math.ceil(n_tokens / (bs * self.bptt))
        t_size = n_batches * self.bptt * bs
        buffer = np.zeros(t_size, dtype=self.sents.dtype) + self.eos_index
        buffer[t_size - n_tokens:] = self.sents
        buffer = buffer.reshape((bs, n_batches * self.bptt)).T
        self.data = np.zeros((n_batches * self.bptt + 1, bs),
                             dtype=self.sents.dtype) + self.eos_index
        self.data[1:] = buffer
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        return self.data[idx * self.bptt:(idx + 1) * self.bptt].astype(
            np.int64), [self.lang_id] * self.batch_size

    def collate_fn(self, data):
        sents, langids = data[0][0], data[0][1]
        sents = sents.T
        sents = torch.from_numpy(sents.astype(np.int64))
        langids = torch.tensor(langids, dtype=torch.long)
        return sents, langids  # [bsz, 1]


class MonolingualBinaryDataSet(data.Dataset):
    def __init__(self, items, data_config, src_vocab):
        assert items['word2id'] == src_vocab.stoi
        self.pos = items['positions']
        self.sents = items['sents']
        self.lang = items['lang']
        self.lang_id = src_vocab.index(f"[[{self.lang}]]")
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.pad_index = src_vocab.pad_index
        self.eos_index = src_vocab.eos_index
        if self.lang_id == src_vocab.unk_index:
            logger.info("[lang_id] convert into [unk]")

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, index):
        a, b = self.pos[index]
        return self.sents[a:b].astype(np.int64), self.lang_id

    def collate_fn(self, data):
        sents = [d[0] for d in data]
        langids = [d[1] for d in data]
        sents = batch_data(sents, self.pad_index, self.eos_index)
        langids = torch.tensor(langids, dtype=torch.long)
        return sents, langids  # [bsz, 1]


class MonolingualTextDataSet(BaseTextDataSet):
    def __init__(self, items, data_config, src_vocab):

        super().__init__(items, data_config, src_vocab)
        self.items = items
        self.feature = data_config['feature']
        self.src_vocab = src_vocab
        self.max_seq_length = data_config.get("max_seq_length", 512)
        self.data_flow = self._build_process_flow()
        self.pad_index = src_vocab.pad_index
        self.eos_index = src_vocab.eos_index
        self.check_data()

    def _build_process_flow(self):
        def lang_encode(lang):
            return self.src_vocab.index(f"[{lang}_embed]", no_unk=True)

        return {
            'seq1': [self.src_vocab.encode],
            'lang1': [lang_encode],
        }

    def collate_fn(self, data):
        batch, bsz = [], len(data)
        for i, feat in enumerate(self.feature):
            example = []
            for j in range(bsz):
                example.append(data[j][i])
            if feat.startswith('seq'):
                example = batch_data(example, self.pad_index, self.eos_index)
            else:
                example = torch.tensor(example, dtype=torch.long)
            batch.append(example)
        return batch
