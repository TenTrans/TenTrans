import os
import logging
import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.nn as nn
from src.data.vocab import Vocabulary
from src.models.encoder.sentence_rep import SentenceRepModel
from src.models.seq2seq_model import Seq2SeqModel
from src.utils.utility import to_cuda, truncate, add_bert_mask, distributed_model, get_model, score_bleu, bpe_postprocess
from src.data.dataset.paired_dataset import PairedBinaryDataSet
from src.data.sampler import BucketBatchSampler
from src.data.reader import BinaryDataReader
from src.tasks.base_task import BaseTask
from src.loss.labelsmooth_cross_entropy import LabelSmoothingCrossEntropy
from src.search.greedy_search import greedy_search

logger = logging.getLogger(__name__)
device = None
TASK_NAME = "seq2seq"


class Seq2SeqTask(BaseTask):
    def __init__(self,
                 data,
                 model,
                 config,
                 vocab,
                 tgt_vocab,
                 task_id="",
                 local_rank=None):
        super().__init__(data, model, config, vocab, task_id, local_rank)
        self.eos_index = vocab.eos_index
        self.bos_indeex = vocab.bos_index
        self.pad_index = vocab.pad_index
        self.tgt_vocab = tgt_vocab
        self.loss_fn = LabelSmoothingCrossEntropy(len(tgt_vocab),
                                                  config['label_smoothing'])
        self.bpe_type = config.get('bpe_type', 'subword')
        self.task_name = TASK_NAME
        if self.local_rank == 0 or self.local_rank == None:
            logger.info(self.model)

    def train_step(self):
        self.model.train()
        batch = self.get_batch('train')
        src, lang1_id, tgt, lang2_id = batch
        src = truncate(src, self.max_seq_length, self.pad_index,
                       self.eos_index)
        tgt = truncate(tgt, self.max_seq_length, self.pad_index,
                       self.eos_index)
        y = tgt[:, :-1]
        y_label = tgt[:, 1:]
        src, lang1_id, y, lang2_id, y_label = to_cuda(src, lang1_id, y,
                                                      lang2_id, y_label)
        no_pad_mask = y_label != self.pad_index
        tensor = self.model('fwd',
                            src=src,
                            tgt=y,
                            lang1_id=lang1_id,
                            lang2_id=lang2_id)

        logits = self.model('predict', tensor=tensor[no_pad_mask])
        y_label = y_label[no_pad_mask]
        loss = self.loss(logits, y_label)
        return loss

    def loss(self, logits, label):
        return self.loss_fn(logits, label)

    def eval_step(self):
        self.model.eval()
        valid_score = 0
        hyp_list = []
        tgt_list = []
        with torch.no_grad():
            for name in ['valid', 'test']:
                total_loss = 0
                total_num = 0
                for i, batch in enumerate(iter(self.data[name])):
                    src, lang1_id, tgt, lang2_id = batch
                    src = truncate(src, self.max_seq_length, self.pad_index,
                                   self.eos_index)
                    tgt = truncate(tgt, self.max_seq_length, self.pad_index,
                                   self.eos_index)

                    y = tgt[:, :-1]
                    y_label = tgt[:, 1:]

                    src, lang1_id, y, lang2_id, y_label = to_cuda(
                        src, lang1_id, y, lang2_id, y_label)
                    no_pad_mask = y_label != self.pad_index
                    tensor = self.model('fwd',
                                        src=src,
                                        tgt=y,
                                        lang1_id=lang1_id,
                                        lang2_id=lang2_id)
                    logits = self.model('predict', tensor=tensor[no_pad_mask])
                    y_label = y_label[no_pad_mask]
                    loss = self.loss(logits, y_label)
                    total_loss += loss.item() * y_label.size(-1)
                    total_num += y_label.size(-1)

                    hyp = greedy_search(get_model(self.model), src, lang1_id,
                                        lang2_id)
                    for j in range(len(y)):
                        tgt_txt = " ".join(
                            self.tgt_vocab.decode(tgt[j].tolist(),
                                                  no_special=True))
                        hyp_txt = " ".join(
                            self.tgt_vocab.decode(hyp[j].tolist(),
                                                  no_special=True))
                        hyp_list.append(
                            bpe_postprocess(hyp_txt, bpe_type=self.bpe_type))
                        tgt_list.append(
                            bpe_postprocess(tgt_txt, bpe_type=self.bpe_type))

                bleu = score_bleu(hyp_list, tgt_list)
                loss = total_loss / total_num

                if self.local_rank == 0 or self.local_rank == None:
                    logger.info(
                        "Epoch_{}  Result Task_{}.{} on {}, loss {:5.2f}, ppl {:5.2f}, bleu {:5.3f},ntokens {:5d}"
                        .format(self.epoch ,self.task_name, self.task_id, name, loss,
                                2**loss, bleu, total_num))
                if name == 'valid':
                    valid_score = -loss

        return valid_score

    @classmethod
    def load_data(cls, data_config, src_vocab, tgt_vocab, multi_gpu):

        dataloader = {}
        names = ['train', 'valid', 'test']

        for i, filename in enumerate(data_config['train_valid_test']):
            name = names[i]
            if data_config.get('split_data', False) and name == 'train':
                assert multi_gpu
                filename_ = []
                for fn in filename.split(":"):
                    fn = f"{fn}.{dist.get_rank()}"
                    filename_.append(fn)
                filename = ":".join(filename_)

            inputItems = BinaryDataReader(
                data_config['data_folder']).getInputItems(filename)

            dataset = PairedBinaryDataSet(
                inputItems,
                data_config,
                src_vocab,
                tgt_vocab,
                remove_long_sentence=(name == 'train'))

            shuffle, group_by_size = False, False
            if name == 'train':
                shuffle, group_by_size = True, data_config['group_by_size']

            batch_sampler = BucketBatchSampler(
                dataset,
                shuffle=shuffle,
                batch_size=data_config['batch_size'],
                max_tokens=data_config['max_tokens'],
                group_by_size=group_by_size)

            dataloader[name] = data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               collate_fn=dataset.collate_fn,
                                               num_workers=1,
                                               batch_sampler=batch_sampler,
                                               pin_memory=False)
        return dataloader

    @classmethod
    def build_task(cls,
                   task_id,
                   config,
                   model=None,
                   sentence_rep=None,
                   local_rank=None):
                   
        data_config = config['data']
        vocab = Vocabulary(file=os.path.join(data_config['data_folder'],
                                             data_config['src_vocab']))
        tgt_vocab = Vocabulary(file=os.path.join(data_config['data_folder'],
                                                 data_config['tgt_vocab']))

        dataloader = cls.load_data(data_config, vocab, tgt_vocab,
                                   config['multi_gpu'])

        if model:
            _model = model
        elif sentence_rep:
            sentenceRepModel = sentence_rep
            _model = Seq2SeqModel(config['target'],
                                  sentenceRepModel,
                                  tgt_vocab=tgt_vocab)
        else:
            sentenceRepModel = SentenceRepModel.build_model(
                config['sentenceRep'], vocab)
            _model = Seq2SeqModel(config['target'],
                                  sentenceRepModel,
                                  tgt_vocab=tgt_vocab)

        if config['multi_gpu']:
            _model = distributed_model(_model, config)
        else:
            _model.cuda()
        return cls(dataloader, _model, config, vocab, tgt_vocab, task_id,
                   local_rank)
