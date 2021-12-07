from src.tasks import task_builder
from src.optimizer.lr_scheduler_builder import lr_sheduler_builder
from src.optimizer.optimizer_builder import optimizer_builder
import logging
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.utils.utility import get_model

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext

logger = logging.getLogger(__name__)


class TaskManger:
    def __init__(self, config, local_rank):
        self.config = config
        self.step = 0
        self.local_rank = local_rank

        self.update_every_epoch = config["update_every_epoch"]
        self.save_interval = config["save_interval"]

        self.share_all_task_model = config["share_all_task_model"]
        self.share_all_task_sentence_rep = config[
            "share_all_task_sentence_rep"]

        self.epoch = config["epoch"]
        self.dump_dir = config["dumpdir"]

        self.log_interval = config["log_interval"]
        self.accumulate_gradients = config["accumulate_gradients"]
        self.multi_task_mode = config["multi_task_mode"]

        self.tasks = self.build_tasks()
        self.tasks_set = {
            f"{task.task_name}.{task.task_id}": task
            for task in self.tasks
        }
        self.log = {
            f"{task.task_name}.{task.task_id}": []
            for task in self.tasks
        }

        self.optimizer = self.build_optimizer()
        self.scheduler, self.scheduler_step_at = self.build_scheduler()

        self.build_task_optimizer()
        self.bulid_task_scheduler()
        self.reload_task_checkpoint()

    def train(self):
        epoch = self.tasks[0].epoch
        while epoch < self.epoch:
            for step in range(self.update_every_epoch):
                self.train_step()
                self.log_info(epoch,
                              step + epoch * self.update_every_epoch + 1)

            self.eval_step()
            epoch += 1
            for task in self.tasks:
                task.set_epoch(epoch)
            if epoch % self.save_interval == 0:
                self.save_task_checkpoint(epoch)
            self.save_task_checkpoint("last")
            self.is_stop()

    def optimize(self, task, loss):

        if self.accumulate_gradients == 1:
            loss.backward()
            if task.clip_grad_norm > 0:
                clip_grad_norm_(task.model.parameters(), task.clip_grad_norm)
            task.optimizer.step()
            task.optimizer.zero_grad()
            if task.scheduler and task.scheduler_step_at == "step":
                task.scheduler.step()

        elif self.accumulate_gradients > 1:
            sync_context = (task.model.no_sync if self.local_rank is not None
                            and self.step % self.accumulate_gradients != 0
                            and self.config["NPROC_PER_NODE"] > 1 else
                            nullcontext)
            with sync_context():
                loss = loss / self.accumulate_gradients
                loss.backward()

            if self.step % self.accumulate_gradients == 0:
                task.optimizer.step()
                task.optimizer.zero_grad()
                if task.scheduler and task.scheduler_step_at == "step":
                    task.scheduler.step()

    def multi_task_optimize(self, task, loss, cur_idx, num_task):
        if cur_idx < num_task - 1:
            sync_context = (task.model.no_sync
                            if self.local_rank is not None else nullcontext)
            with sync_context():
                loss.backward()
        else:
            self.optimize(task, loss)

    def is_stop(self):
        for task in self.tasks:
            if self.local_rank is None or self.local_rank == 0:
                task.stop_training()

    def log_info(self, epoch, step):
        if step % self.log_interval != 0:
            return
        log = {}
        for k, v in self.log.items():
            if len(v) != 0:
                log[k] = sum(v) / len(v)

        for k in self.log:
            self.log[k] = []

        s = f"epoch_{epoch} - step_{step}:"
        for k, v in log.items():
            s = (s + f" - {k} loss:{v:.2f}" + " - lr:{:.2e}".format(
                self.tasks_set[k].optimizer.param_groups[0]["lr"]))

        logger.info(s)

    def build_tasks(self):
        model, rep, tasks = None, None, []
        for _, (task_id,
                task_params) in enumerate(self.config["tasks"].items()):

            task_name = task_params["task_name"]
            task = task_builder[task_name].build_task(task_id, task_params,
                                                      model, rep,
                                                      self.local_rank)
            if self.share_all_task_model:
                #model = get_model(task.model)
                model = task.model
                
            if self.share_all_task_sentence_rep:
                m = get_model(task.model)
                rep = m.sentenceRep
            tasks.append(task)
        return tasks

    def train_step(self):
        task_weights = np.array([task.task_weight for task in self.tasks])
        if (task_weights == task_weights[0]).all():
            tasks = self.tasks
        else:
            task_weights = task_weights / task_weights.sum()
            tasks = np.random.choice(self.tasks,
                                     min(len(self.tasks), 5),
                                     replace=True,
                                     p=task_weights)

        for idx, task in enumerate(tasks):
            loss = task.train_step()
            self.step += 1
            if self.multi_task_mode:
                self.multi_task_optimize(task, loss, idx, len(tasks))
            else:
                self.optimize(task, loss)
            self.log[f"{task.task_name}.{task.task_id}"].append(loss)

    def eval_step(self):
        for task in self.tasks:
            score = task.eval_step()
            if task.best_score < score:
                task.best_score = score
                task.stop_patience = 0
                self.save_task_best_checkpoint(task)
            else:
                task.stop_patience += 1

    def build_optimizer(self):
        share_optimizer = self.config.get("optimizer", None)
        if not share_optimizer:
            return None

        if self.share_all_task_model:
            parameters = self.tasks[0].model.parameters()
        else:
            parameters = []
            for task in self.tasks:
                parameters += list(task.model.parameters())
        optimizer = optimizer_builder(self.config, parameters)
        return optimizer

    def build_scheduler(self):
        share_scheduler = self.config.get("scheduling", False)
        if not share_scheduler:
            return None, None
        scheduler, scheduler_step_at = lr_sheduler_builder(
            self.config, self.optimizer)
        return scheduler, scheduler_step_at

    def build_task_optimizer(self):
        for task in self.tasks:
            task.build_optimizer(self.optimizer)

    def bulid_task_scheduler(self):
        for task in self.tasks:
            task.build_scheduler(self.scheduler, self.scheduler_step_at)

    def save_task_checkpoint(self, name):

        if self.share_all_task_model:
            self.tasks[0].save_checkpoint(self.dump_dir, name)
        else:
            for task in self.tasks:
                task.save_checkpoint(self.dump_dir, name)

    def save_task_best_checkpoint(self, task):
        if self.share_all_task_model:
            self.tasks[0].save_best_checkpoint(self.dump_dir)
        else:
            task.save_best_checkpoint(self.dump_dir)

    def reload_task_checkpoint(self):
        for task in self.tasks:
            task.reload_checkpoint(self.dump_dir)
