import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
from src.data.vocab import Vocabulary
from src.models.encoder.sentence_rep import SentenceRepModel
from src.models.classification_model import ClassificationModel
from src.tasks.base_task import BaseTask
from src.utils.utility import (
    to_cuda,
    accuracy,
    f1_recall_precision,
    truncate,
    distributed_model,
)
from src.optimizer.optimizer_builder import Adam
from src.data.reader import CSVDataReader
from src.data.dataset.classification_dataset import ClassificationTextDataSet
from src.data.sampler import BucketBatchSampler

logger = logging.getLogger(__name__)
TASK_NAME = "classification"


class ClassificationTask(BaseTask):
    def __init__(self,
                 data,
                 model,
                 config,
                 vocab,
                 task_id="",
                 local_rank=None):
        super().__init__(data, model, config, vocab, task_id, local_rank)
        self.weight_training = config["weight_training"]
        self.task_name = TASK_NAME
        if self.local_rank == 0 or self.local_rank is None:
            logger.info(self.model)

    def train_step(self):
        self.model.train()
        batch = self.get_batch("train")
        text, langid, label = batch
        text = truncate(text, self.max_seq_length, self.vocab.pad_index,
                        self.vocab.eos_index)
        text, langid, label = to_cuda(text, langid, label)
        logits = self.model(text, langid)
        loss = self.loss(logits, label)
        return loss

    def loss(self, logits, label):
        if self.weight_training:
            if not hasattr(self, "weights"):
                labels = []
                for batch in self.data["train"]:
                    labels.extend(batch[2].tolist())
                labels = torch.LongTensor(labels)
                weights = torch.FloatTensor([
                    1.0 / (labels == i).sum().float()
                    for i in range(logits.size(1))
                ]).cuda()
                weights = weights / weights.sum().cuda()
                setattr(self, "weights", weights)
        else:
            setattr(self, "weights", None)
        return F.cross_entropy(logits, label, self.weights)

    def eval_step(self):
        self.model.eval()
        with torch.no_grad():
            for name in ["valid", "test"]:
                golden = []
                predicts = []
                for _, batch in enumerate(iter(self.data[name])):
                    text, langid, label = batch
                    text = truncate(
                        text,
                        self.max_seq_length,
                        self.vocab.pad_index,
                        self.vocab.bos_index,
                    )
                    golden.extend(label.cpu().tolist())
                    text, langid, label = to_cuda(text, langid, label)
                    logits = self.model(text, langid)
                    _, predict = logits.topk(k=1, dim=-1)
                    predicts.extend(predict.view(-1).cpu().tolist())

                accu = accuracy(golden, predicts)
                result = f1_recall_precision(golden, predicts)
                if self.local_rank == 0 or self.local_rank is None:
                    logger.info("Result of Task_{}.{} on {}, {}".format(
                        self.task_name, self.task_id, name, result))
                    logger.info(
                        "Result of Task_{}.{} on {}, acc:{:.3f}".format(
                            self.task_name, self.task_id, name, accu))

                if name == "valid":
                    valid_score = result[1]["f1"]
        return valid_score

    def build_optimizer(self, optimizer=None, share_all_task_model=False):

        multi_gpu = True if hasattr(self.model, "module") else False
        optimizer = Adam(
            self.model.sentence_rep.parameters()
            if not multi_gpu else self.model.module.sentence_rep.parameters(),
            self.config["lr_e"],
        )
        optimizer = Adam(
            [
                {
                    "params":
                    self.model.sentence_rep.parameters() if not multi_gpu else
                    self.model.module.sentence_rep.parameters(),
                    "lr":
                    self.config["lr_e"],
                },
                {
                    "params":
                    self.model.target.parameters() if not multi_gpu else
                    self.model.module.target.parameters(),
                    "lr":
                    self.config["lr_p"],
                },
            ],
            self.config["lr_p"],
        )
        setattr(self, "optimizer", optimizer)

    @classmethod
    def load_data(cls, data_config, src_vocab, multi_gpu):
        dataloader = {}
        names = ["train", "valid", "test"]

        for i, filename in enumerate(data_config["train_valid_test"]):
            name = names[i]
            inputitems = CSVDataReader(
                data_config["data_folder"]).get_input_items(filename)
            dataset = ClassificationTextDataSet(inputitems, data_config,
                                                src_vocab)

            num_workers = 20
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
        vocab = Vocabulary(
            file=os.path.join(data_config["data_folder"],
                              data_config["src_vocab"]),
            max_vocab=data_config.get("max_vocab", -1),
        )
        dataloader = cls.load_data(data_config, vocab, config["multi_gpu"])
        if model:
            _model = model
        elif sentence_rep:
            sentence_rep_model = sentence_rep
            _model = ClassificationModel(config["target"], sentence_rep_model,
                                         vocab)
        else:
            if config["sentenceRep"].get("pretrain_rep", None):

                path = config["sentenceRep"]["pretrain_rep"]
                state = torch.load(path, map_location="cpu")
                pretrain_config = state["config"]
                sentence_rep_model = SentenceRepModel.build_model(
                    pretrain_config["sentenceRep"], vocab)
                sentence_rep_model.load_state_dict(state["model_sentenceRep"])
                config["sentenceRep"] = pretrain_config["sentenceRep"]
            else:
                sentence_rep_model = SentenceRepModel.build_model(
                    config["sentenceRep"], vocab)
            _model = ClassificationModel(config["target"], sentence_rep_model,
                                         vocab)

        if config["multi_gpu"] and not hasattr(model, "module"):
            _model = distributed_model(_model, config)
        else:
            _model.cuda()
        return cls(dataloader, _model, config, vocab, task_id, local_rank)
