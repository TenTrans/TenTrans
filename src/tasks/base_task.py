import logging
import os
from src.optimizer.lr_scheduler_builder import lr_sheduler_builder
from src.optimizer.optimizer_builder import optimizer_builder
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class BaseTask:
    def __init__(self,
                 data,
                 model,
                 config,
                 vocab,
                 task_id="",
                 local_rank=None):
        self.data = data
        self.model = model
        self.config = config
        self.vocab = vocab
        self.task_id = task_id
        self.dataiter = self.get_iterator()
        self._epoch = 0
        self.epoch = 0
        self.stop_patience = 0
        self.local_rank = local_rank
        self.best_score = float("-inf")
        self.max_seq_length = config["data"]["max_seq_length"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.patience = config["patience"]
        self.task_weight = config["task_weight"]
        self.reset_optimizer = config["reset_optimizer"]
        logger.info("Number of model parameters: %i" % sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]))

    def stop_training(self):
        if self.stop_patience >= self.patience:
            logger.info(f"Stopping training since no better result occurs in task[{self.task_name}_{self.task_id}].")
            exit()
        elif self.stop_patience != 0:
            logger.info(f"No better result occurs in Task_{self.task_name}_{self.task_id}\
                    ({self.stop_patience}/{self.patience}).")
        else:
            logger.info(
                f"Best Result of Task_{self.task_name}_{self.task_id} on valid: {abs(self.best_score):.3f}")

    def train_step(self):
        # self.n_iter += 1
        raise NotImplementedError

    def loss(self, logits, label):
        raise NotImplementedError

    def eval_step(self):
        raise NotImplementedError

    def set_epoch(self, epoch):
        self.epoch = epoch

    def save_checkpoint(self, path, name=""):
        model = self.model.module if self.config.get("multi_gpu",
                                                     False) else self.model
        if self.config.get("multi_gpu", False) and dist.get_rank() != 0:
            return
        state = {
            "model_sentenceRep": model.sentence_rep.state_dict(),
            "model_target": model.target.state_dict(),
            "epoch": self.epoch,  # for task_manger
            "_epoch": self._epoch,  # for specific task
            "word2id": dict(self.vocab.stoi),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "best_score": self.best_score,
        }
        filename = "checkpoint_{}_{}_{}".format(self.task_name, self.task_id,
                                                name)
        logger.info(f"saving model at {os.path.join(path, filename)}...")
        torch.save(state, os.path.join(path, filename))

        if self.config["keep_last_checkpoint"] > 0 and type(name) == int:
            keep_last_checkpoint = self.config["keep_last_checkpoint"]
            bound = name - keep_last_checkpoint * self.config["save_interval"]
            filename = "checkpoint_{}_{}_{}".format(self.task_name,
                                                    self.task_id, bound)
            if bound > 0 and os.path.lexists(os.path.join(path, filename)):
                os.remove(os.path.join(path, filename))

    def save_best_checkpoint(self, path):
        self.save_checkpoint(path, "best")

    def reload_checkpoint(self, path):

        checkpoint_last_path = os.path.join(
            path,
            "checkpoint_{}_{}_last".format(
                self.task_name,
                self.task_id,
            ),
        )

        if not os.path.lexists(checkpoint_last_path):
            checkpoint_last_path = ""

        reload_path = (self.config["reload_checkpoint"]
                       if self.config["reload_checkpoint"] else
                       checkpoint_last_path)

        if not reload_path:
            return

        logger.info("Reloaded model from {}...".format(reload_path))
        state = torch.load(reload_path, map_location="cpu")

        if not self.reset_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self._epoch = state["_epoch"]

        if self.config.get("multi_gpu", False):
            self.model.module.sentence_rep.load_state_dict(
                state["model_sentenceRep"])
            self.model.module.target.load_state_dict(state["model_target"])
        else:
            self.model.sentence_rep.load_state_dict(state["model_sentenceRep"])
            self.model.target.load_state_dict(state["model_target"])

        self.best_score = state.get("best_score", float("-inf"))
        # self.optimizer.set_lr(self.scheduler.compute_rate())

    @classmethod
    def build_task(cls, config):
        raise NotImplementedError

    def get_iterator(self):
        dataiter = {}
        for name in ["train", "valid", "test"]:
            dataiter[name] = iter(self.data[name])
        return dataiter

    def get_batch(self, name):
        try:
            batch = next(self.dataiter[name])
        except StopIteration:
            if name == 'train':
                logger.info(f"{self.task_id} new epoch.")
                self._epoch += 1
                if not getattr(self, 'tgt_vocab', None):
                    self.data = self.load_data(self.config['data'], self.vocab, self.config.get("multi_gpu", False))
                else:
                    self.data = self.load_data(self.config['data'], self.vocab, self.tgt_vocab, self.config.get("multi_gpu", False))
                self.dataiter = self.get_iterator()
            self.dataiter[name] = iter(self.data[name])
            batch = next(self.dataiter[name])
        return batch

    def build_optimizer(self, optimizer):
        if not optimizer:
            optimizer = optimizer_builder(self.config, self.model.parameters())
        setattr(self, "optimizer", optimizer)

    def build_scheduler(
        self,
        scheduler=None,
        scheduler_step_at=None,
    ):

        if not scheduler:
            scheduler, scheduler_step_at = lr_sheduler_builder(
                self.config, self.optimizer)
        setattr(self, "scheduler", scheduler)
        setattr(self, "scheduler_step_at", scheduler_step_at)
