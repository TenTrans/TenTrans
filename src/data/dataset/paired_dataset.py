import torch
import torch.utils.data as data
from src.utils.utility import batch_data
from .dataset import BaseTextDataSet
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PairedBinaryDataSet(data.Dataset):
    def __init__(self,
                 items,
                 data_config,
                 src_vocab,
                 tgt_vocab,
                 remove_long_sentence=False):
        items1, items2 = items
        self.pos1 = items1['positions']
        self.sents1 = items1['sents']
        self.lang1 = items1['lang']
        self.lang1_id = src_vocab.index(f"[[{self.lang1}]]")
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]

        self.pos2 = items2['positions']
        self.sents2 = items2['sents']
        self.lang2 = items2['lang']
        self.lang2_id = src_vocab.index(f"[[{self.lang2}]]")
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        self.pad_index = src_vocab.pad_index
        self.eos_index = src_vocab.eos_index
        self.lengths = self.lengths1 + self.lengths2 + 4

        if self.lang1_id == src_vocab.unk_index:
            logger.info(f"[{self.lang1}] convert into [unk]")

        if self.lang2_id == src_vocab.unk_index:
            logger.info(f"[{self.lang2}] convert into [unk]")

        assert self.pad_index == tgt_vocab.pad_index
        assert self.eos_index == tgt_vocab.eos_index
        assert len(self.pos1) == len(self.pos2)
        assert self.sents1.max() < len(src_vocab)
        assert self.sents2.max() < len(tgt_vocab)

        assert items1['word2id'] == src_vocab.stoi
        assert items2['word2id'] == tgt_vocab.stoi

        if remove_long_sentence:
            self.remove_long_sentences(data_config.get('max_len', 100))
        logger.info("Load %i sentences" % len(self.pos1))

    def remove_long_sentences(self, max_len):
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.lengths = self.lengths1 + self.lengths2
        logger.info("Removed %i too long sentences." %
                    (init_size - len(indices)))

    def __len__(self):
        return len(self.pos1)

    def __getitem__(self, index):
        a1, b1 = self.pos1[index]
        a2, b2 = self.pos2[index]
        return self.sents1[a1:b1].astype(
            np.int64), self.lang1_id, self.sents2[a2:b2].astype(
                np.int64), self.lang2_id

    def collate_fn(self, data):
        bsz = len(data)
        sents1 = [d[0].tolist() for d in data]
        lang1_id = [d[1] for d in data]

        sents2 = [d[2].tolist() for d in data]
        lang2_id = [d[3] for d in data]

        sents1 = batch_data(sents1, self.pad_index, self.eos_index)
        lang1_id = torch.tensor(lang1_id, dtype=torch.long)  # bsz x 1
        lang1_id = lang1_id.view(bsz, -1).repeat(1, sents1.size(1))

        sents2 = batch_data(sents2, self.pad_index, self.eos_index)
        lang2_id = torch.tensor(lang2_id, dtype=torch.long)
        lang2_id = lang2_id.view(bsz, -1).repeat(1, sents2.size(1))

        return sents1, lang1_id, sents2, lang2_id
