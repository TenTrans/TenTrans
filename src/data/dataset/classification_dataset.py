import torch
from src.utils.utility import batch_data
from .dataset import BaseTextDataSet


class ClassificationTextDataSet(BaseTextDataSet):
    def __init__(self, items, data_config, src_vocab, tgt_vocab=None, name=""):

        super().__init__(items, data_config, src_vocab)
        self.items = items
        self.feature = data_config["feature"]
        self.src_vocab = src_vocab
        self.max_seq_length = data_config.get("max_seq_length", 512)
        self.label12id = data_config["label12id"]
        self.data_flow = self._build_process_flow()
        self.pad_index = src_vocab.pad_index
        self.eos_index = src_vocab.eos_index  # bos == eos
        assert set(self.items[0].content.keys()) == set(self.feature)
        self.check_data()

    def _build_process_flow(self):
        def lang2id(lang):
            return self.src_vocab.index(f"[[{lang}]]")

        return {
            "seq1": [self.src_vocab.encode],
            "lang1": [lang2id],
            "label1": [self.label12id.__getitem__],
        }

    def collate_fn(self, data):
        batch, bsz = [], len(data)
        for i, feat in enumerate(self.feature):
            example = []
            for j in range(bsz):
                example.append(data[j][i])
            if feat.startswith("seq"):
                example = batch_data(example, self.pad_index, self.eos_index)
            else:
                example = torch.tensor(example, dtype=torch.long)
            batch.append(example)
        return batch
