import os
import logging
import random
from src.data.vocab import Vocabulary
from src.models.encoder.sentence_rep import SentenceRepModel
from src.models.masked_language_model import MaskLanguageModel
from src.utils.utility import (
    to_cuda,
    truncate,
    add_bert_mask,
    concate,
    distributed_model,
)
from src.data.dataset.paired_dataset import PairedBinaryDataSet
from src.data.sampler import BucketBatchSampler
from src.data.reader import BinaryDataReader
import torch.nn as nn
import torch
import torch.utils.data as data
import torch.distributed as dist
from .mlm_task import MLMTask

logger = logging.getLogger(__name__)
TASK_NAME = "tlm"


class TLMTask(MLMTask):
    def __init__(self,
                 data,
                 model,
                 config,
                 vocab,
                 task_id="",
                 local_rank=None):

        super().__init__(data, model, config, vocab, task_id)
        self.p_pred, self.p_mask, self.p_keep, self.p_rand = config.get(
            "p_pred_mask_kepp_rand", [0.15, 0.80, 0.10, 0.10])
        self.mask_index = self.vocab.index("[[mask]]", no_unk=True)
        self.task_name = TASK_NAME
        if self.local_rank == 0 or self.local_rank is None:
            logger.info(self.model)
        self.tgt_vocab = vocab
        
    def train_step(self):

        self.model.train()
        batch = self.get_batch("train")
        src, lang1_id, tgt, lang2_id = batch

        x, langid, positions = concate(
            src,
            tgt,
            lang1_id,
            lang2_id,
            self.vocab.pad_index,
            self.vocab.eos_index,
            reset_positions=True,
        )
        x = truncate(x, self.max_seq_length, self.vocab.pad_index,
                     self.vocab.eos_index)
        langid = truncate(langid, self.max_seq_length, -1, -1)
        positions = truncate(positions, self.max_seq_length, -1, -1)
        x, pred_mask, label = add_bert_mask(
            x,
            self.vocab.pad_index,
            self.mask_index,
            len(self.vocab),
            self.p_pred,
            self.p_mask,
            self.p_keep,
            self.p_rand,
        )

        x, langid, label, pred_mask, positions = to_cuda(
            x, langid, label, pred_mask, positions)
        tensor = self.model(x, langid, pred_mask, positions=positions)
        loss = self.loss(tensor, label)

        return loss

    def loss(self, logits, label):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, label)

    def eval_step(self):
        self.model.eval()
        valid_score = 0
        with torch.no_grad():
            for name in ["valid", "test"]:
                total_loss = 0
                total_num = 0
                for _, batch in enumerate(iter(self.data[name])):
                    src, lang1_id, tgt, lang2_id = batch
                    x, langid, positions = concate(
                        src,
                        tgt,
                        lang1_id,
                        lang2_id,
                        self.vocab.pad_index,
                        self.vocab.eos_index,
                        reset_positions=True,
                    )
                    x = truncate(
                        x,
                        self.max_seq_length,
                        self.vocab.pad_index,
                        self.vocab.eos_index,
                    )
                    langid = truncate(langid, self.max_seq_length, -1, -1)
                    positions = truncate(positions, self.max_seq_length, -1,
                                         -1)
                    x, pred_mask, label = add_bert_mask(
                        x,
                        self.vocab.pad_index,
                        self.mask_index,
                        len(self.vocab),
                        self.p_pred,
                        self.p_mask,
                        self.p_keep,
                        self.p_rand,
                        fix=True,
                    )
                    x, langid, label, pred_mask, positions = to_cuda(
                        x, langid, label, pred_mask, positions)
                    tensor = self.model(src=x,
                                        lang_id=langid,
                                        pred_mask=pred_mask,
                                        positions=positions)
                    loss = self.loss(tensor, label)

                    total_num += label.size(-1)
                    total_loss += loss.item() * label.size(-1)

                loss = total_loss / total_num
                if self.local_rank == 0 or self.local_rank is None:
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
    def load_data(cls, data_config, src_vocab, tgt_vocab, multi_gpu):

        dataloader = {}
        names = ["train", "valid", "test"]

        for i, filename in enumerate(data_config["train_valid_test"]):
            name = names[i]
            split_data = data_config.get("split_data", 1)
            idx = random.randint(0, split_data-1)
            if split_data > 1 and name == "train":
                filename_ = []
                for fn in filename.split(":"):
                    fn = f"{fn}.{idx}"
                    filename_.append(fn)
                filename = ":".join(filename_)

            inputitems = BinaryDataReader(
                data_config["data_folder"]).get_input_items(filename)

            dataset = PairedBinaryDataSet(
                inputitems,
                data_config,
                src_vocab,
                tgt_vocab,
                remove_long_sentence=(name == "train"),
            )

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
                num_workers=1,
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

        dataloader = cls.load_data(data_config, vocab, vocab,
                                   config["multi_gpu"])
        if model:
            _model = model
        elif sentence_rep:
            sentence_rep_model = sentence_rep
            _model = MaskLanguageModel(config["target"], sentence_rep_model,
                                       vocab)
        else:
            sentence_rep_model = SentenceRepModel.build_model(
                config["sentenceRep"], vocab)
            _model = MaskLanguageModel(config["target"], sentence_rep_model,
                                       vocab)

        if config["multi_gpu"] and not hasattr(model, "module"):
            _model = distributed_model(_model, config)
        else:
            _model.cuda()

        return cls(dataloader, _model, config, vocab, task_id, local_rank)
