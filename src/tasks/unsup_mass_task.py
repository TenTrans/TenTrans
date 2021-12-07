import logging
import os
import numpy as np
from src.data.vocab import Vocabulary
from src.models.seq2seq_model import Seq2SeqModel
from src.utils.utility import to_cuda, truncate, distributed_model
from src.data.dataset.monolingual_dataset import MonolingualBinaryDataSet
from src.data.sampler import BucketBatchSampler
from src.data.reader import BinaryDataReader
from src.tasks.base_task import BaseTask
from src.loss.labelsmooth_cross_entropy import LabelSmoothingCrossEntropy
import torch
import torch.utils.data as data
import torch.distributed as dist

logger = logging.getLogger(__name__)
TASK_NAME = "unsup_mass"


class UnsuperMassTask(BaseTask):
    def __init__(self,
                 data,
                 model,
                 config,
                 vocab,
                 tgt_vocab,
                 task_id="",
                 local_rank=None):
        super().__init__(data, model, config, vocab, task_id)
        self.eos_index = vocab.eos_index
        self.bos_indeex = vocab.bos_index
        self.pad_index = vocab.pad_index
        self.mask_index = self.vocab.index("[[mask]]", no_unk=True)
        self.tgt_vocab = tgt_vocab
        self.loss_fn = LabelSmoothingCrossEntropy(len(tgt_vocab),
                                                  config["label_smoothing"])
        self.ratio = 0.3
        self.pred_words = torch.FloatTensor([0.1, 0.1, 0.8])
        self.task_name = TASK_NAME
        if self.local_rank == 0 or self.local_rank is None:
            logger.info(self.model)

    def mask_start(self, end):
        p = torch.rand(len(end))
        mask_1 = p >= 0.8
        mask_end = (p >= 0.6) & (p < 0.8)

        start = torch.LongTensor(
            [np.random.randint(1, end[i]) for i in range(len(end))])
        start[mask_1] = 1
        start[mask_end] = end[mask_end]
        return start

    def mask_interval(self, length):
        mask_length = torch.round(length * self.ratio).int()
        mask_length[mask_length < 1] = 1
        mask_start = self.mask_start(length - mask_length)
        return mask_start, mask_length

    def random_word(self, w, pred_probs):
        cands = [self.mask_index, np.random.randint(len(self.vocab)), w]
        prob = torch.multinomial(pred_probs, 1, replacement=True)
        return cands[prob]

    def mass_mask(self, x):
        def _batch_data(data, pad_index):
            lengths = [len(d) for d in data]
            tensor = torch.LongTensor(len(data), max(lengths)).fill_(pad_index)
            for i, _ in enumerate(data):
                tensor[i, 0:lengths[i]].copy_(torch.LongTensor(data[i]))
            return tensor

        lengths = (x != self.pad_index).sum(-1)
        start, mask_length = self.mask_interval(lengths)
        src_input, tgt_input, labels = [], [], []
        for i in range(len(x)):
            output = x[i][start[i]:start[i] + mask_length[i]].tolist()
            _target = x[i][start[i] - 1:start[i] + mask_length[i] - 1].tolist()
            target = []
            for w in _target:
                target.append(self.random_word(w, self.pred_words))
            source = []
            for j, w in enumerate(x[i][:lengths[i]].tolist()):
                if j >= start[i] and j < start[i] + mask_length[i]:
                    w = self.mask_word(w)
                if w is not None:
                    source.append(w)
            src_input.append(source)
            tgt_input.append(target)
            labels.append(output)

        return (
            _batch_data(src_input, self.vocab.pad_index),
            _batch_data(tgt_input, self.vocab.pad_index),
            _batch_data(labels, self.vocab.pad_index),
        )

    def mask_word(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.mask_index
        elif p >= 0.1:
            return np.random.randint(len(self.vocab))
        else:
            return w

    def train_step(self):
        self.model.train()
        batch = self.get_batch("train")
        src, _ = batch
        src = truncate(src, self.max_seq_length, self.pad_index,
                       self.eos_index)
        src, y, y_label = self.mass_mask(src)

        src, y, y_label = to_cuda(src, y, y_label)

        no_pad_mask = y != self.pad_index
        tensor = self.model("fwd",
                            src=src,
                            tgt=y,
                            lang1_id=None,
                            lang2_id=None)

        logits = self.model("predict", tensor=tensor[no_pad_mask])
        y_label = y_label[no_pad_mask]
        loss = self.loss(logits, y_label)
        return loss

    def loss(self, logits, label):
        return self.loss_fn(logits, label)

    def eval_step(self):
        self.model.eval()
        valid_score = 0
        with torch.no_grad():
            for name in ["valid", "test"]:
                total_loss = 0
                total_num = 0
                for _, batch in enumerate(iter(self.data[name])):

                    src, _ = batch
                    src = truncate(src, self.max_seq_length, self.pad_index,
                                   self.eos_index)
                    src, y, y_label = self.mass_mask(src)
                    src, y, y_label = to_cuda(src, y, y_label)

                    no_pad_mask = y_label != self.pad_index
                    tensor = self.model("fwd",
                                        src=src,
                                        tgt=y,
                                        lang1_id=None,
                                        lang2_id=None)
                    logits = self.model("predict", tensor=tensor[no_pad_mask])
                    y_label = y_label[no_pad_mask]
                    loss = self.loss(logits, y_label)
                    total_loss += loss.item() * y_label.size(-1)
                    total_num += y_label.size(-1)

                loss = total_loss / total_num
                if self.local_rank is None or self.local_rank == 0:
                    logger.info(
                        "Result Task_{}.{} on {}, loss {:5.2f}, ppl {:5.2f}, ntokens {:5d}"
                        .format(
                            self.task_name,
                            self.task_id,
                            name,
                            loss,
                            2**loss,
                            total_num,
                        ))
                if name == "valid":
                    valid_score = -loss

        return valid_score

    @classmethod
    def load_data(cls, data_config, src_vocab, multi_gpu):
        dataloader = {}
        names = ["train", "valid", "test"]
        for i, filename in enumerate(data_config["train_valid_test"]):
            name = names[i]
            if data_config.get("split_data", False) and name == "train":
                assert multi_gpu
                filename = f"{filename}.{dist.get_rank()}"

            inputitems = BinaryDataReader(
                data_config["data_folder"]).get_input_items(filename)

            dataset = MonolingualBinaryDataSet(inputitems, data_config,
                                               src_vocab)
            num_workers = 0
            shuffle, group_by_size = False, False
            if name == "train":
                shuffle, group_by_size = True, data_config["group_by_size"]

            batch_sampler = BucketBatchSampler(
                dataset,
                shuffle=shuffle,
                batch_size=data_config["batch_size"],
                max_tokens=data_config["max_tokens"],
                group_by_size=group_by_size,
            )

            dataloader[name] = data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                pin_memory=False,
            )
        return dataloader

    @classmethod
    def build_task(cls,
                   task_id,
                   config,
                   model=None,
                   sentence_rep=None,
                   local_rank=None):
        data_config = config["data"]
        vocab = Vocabulary(file=os.path.join(data_config["data_folder"],
                                             data_config["src_vocab"]))
        tgt_vocab = Vocabulary(file=os.path.join(data_config["data_folder"],
                                                 data_config["tgt_vocab"]))
        dataloader = cls.load_data(data_config, vocab, config["multi_gpu"])

        if model:
            _model = model
        elif sentence_rep:
            sentence_rep_model = sentence_rep
            _model = Seq2SeqModel(config["target"],
                                  sentence_rep_model,
                                  tgt_vocab=tgt_vocab)
        else:
            sentence_rep_model = sentenceRepModel.build_model(
                config["sentenceRep"], vocab)
            _model = Seq2SeqModel(config["target"],
                                  sentence_rep_model,
                                  tgt_vocab=tgt_vocab)

        if config["multi_gpu"] and not hasattr(model, "module"):
            _model = distributed_model(_model, config)
        else:
            _model.cuda()
            
        return cls(dataloader, _model, config, vocab, tgt_vocab, task_id,
                   local_rank)
