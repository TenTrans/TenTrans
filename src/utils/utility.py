from torch import nn
from torch import Tensor
import numpy as np
import torch
import yaml
import random
import os
import shutil
import logging
import errno
import sacrebleu


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parase_config(config):
    config['NPROC_PER_NODE'] = int(
        os.environ['NPROC_PER_NODE']) if 'NPROC_PER_NODE' in os.environ else 1

    config['batch_size'] = config.get('batch_size', 32)
    config['max_tokens'] = config.get('max_tokens', -1)
    config['group_by_size'] = config.get('group_by_size', True)
    config['max_seq_length'] = config.get('max_seq_length', 512)
    config['max_len'] = config.get('max_len', 100)
    config['clip_grad_norm'] = config.get('clip_grad_norm', 5)
    config['share_out_embedd'] = config.get('share_out_embedd', False)
    config['share_all_embedd'] = config.get('share_all_embedd', False)
    config['patience'] = config.get('patience', 5)
    config['label_smoothing'] = config.get('label_smoothing', 0.)
    config['accumulate_gradients'] = config.get('accumulate_gradients', 1)
    config['update_every_epoch'] = config.get('update_every_epoch', 5000)
    config['save_interval'] = config.get('save_interval', 1)
    config['share_all_task_model'] = config.get('share_all_task_model', True)
    config['share_all_task_sentence_rep'] = config.get(
        'share_all_task_sentence_rep', False)
    config['epoch'] = config.get('epoch', 100)
    config['dumpdir'] = config.get('dumpdir', './dump')
    config['log_interval'] = config.get('log_interval', 20)
    config['target'] = config.get('target', {})
    config['sentenceRep'] = config.get('sentenceRep', {})
    config['reset_optimizer'] = config.get('reset_optimizer', False)
    config['keep_last_checkpoint'] = config.get('keep_last_checkpoint', -1)
    config['reload_checkpoint'] = config.get('reload_checkpoint', "")
    config['multi_task_mode'] = config.get('multi_task_mode', False)
    
    for task_id, task_params in config['tasks'].items():
        task_type = task_params['task_name']

        if task_type == 'classification':
            task_params['data']['label12id'] = {
                str(label): i
                for i, label in enumerate(task_params['data'].get(
                    'label1', []))
            }
            task_params['num_label1'] = len(task_params['data']['label12id'])
            task_params['target']['num_label1'] = task_params['num_label1']

        # data setting
        task_params['data']['batch_size'] = task_params['data'].get(
            'batch_size', config['batch_size'])
        task_params['data']['max_tokens'] = task_params['data'].get(
            'max_tokens', config['max_tokens'])
        task_params['data']['max_seq_length'] = task_params['data'].get(
            'max_seq_length', config['max_seq_length'])
        task_params['data']['group_by_size'] = task_params['data'].get(
            'group_by_size', config['group_by_size'])
        task_params['data']['max_len'] = task_params['data'].get(
            'max_len', config['max_len'])
        task_params['data']['NPROC_PER_NODE'] = config['NPROC_PER_NODE']

        #target settings
        task_params['target'] = task_params.get('target', config['target'])
        task_params['target']['share_out_embedd'] = task_params['target'].get(
            'share_out_embedd', config['share_out_embedd'])
        task_params['target']['share_all_embedd'] = task_params['target'].get(
            'share_all_embedd', config['share_all_embedd'])

        task_params['multi_gpu'] = config['multi_gpu']
        task_params['clip_grad_norm'] = task_params.get(
            'clip_grad_norm', config['clip_grad_norm'])

        #training settings
        task_params['patience'] = task_params.get('patience',
                                                  config['patience'])

        task_params['sentenceRep'] = task_params.get('sentenceRep',
                                                     config['sentenceRep'])
        task_params['reset_optimizer'] = task_params.get(
            'reset_optimizer', config['reset_optimizer'])
        task_params['keep_last_checkpoint'] = task_params.get(
            'keep_last_checkpoint', config['keep_last_checkpoint'])
        task_params['save_interval'] = task_params.get('save_interval',
                                                       config['save_interval'])
        task_params['reload_checkpoint'] = task_params.get(
            'reload_checkpoint', config['reload_checkpoint'])

        task_params['NPROC_PER_NODE'] = config['NPROC_PER_NODE']
        task_params['task_weight'] = task_params.get("task_weight", 1)

        #for classification
        task_params['weight_training'] = task_params.get(
            'weight_training', False)

        #for seq2seq
        task_params['label_smoothing'] = task_params.get(
            'label_smoothing', config.get('label_smoothing'))


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def distributed_model(model, config):
    from torch.nn.parallel import DistributedDataParallel as DDP
    local_rank = torch.distributed.get_rank() % config['NPROC_PER_NODE']
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model.to(device)
    model.cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model


def concate(x1, x2, lang1_id, lang2_id, pad_index, eos_index, reset_positions):

    length1 = (x1 != pad_index).sum(dim=-1)
    length2 = (x2 != pad_index).sum(dim=-1)
    length = length1 + length2

    bsz, max_len = length.size(0), length.max()
    x = x1.new(bsz, max_len).fill_(pad_index)
    lang_ids = x1.new(x.size()).fill_(pad_index)

    x[:, :length1.max().item()].copy_(x1)
    lang_ids[:, :length1.max().item()].copy_(lang1_id)
    positions = torch.arange(max_len).unsqueeze(0).repeat(bsz, 1).to(x1.device)

    for i in range(bsz):
        l1 = length1[
            i] if reset_positions else length1 - 1  # reset_positions no eos in the first sentence
        x[i, l1:l1 + length2[i]].copy_(x2[i, :length2[i]])
        lang_ids[i, l1:l1 + length2[i]].copy_(lang2_id[i, :length2[i]])
        if reset_positions:
            positions[i, l1:] -= length1[i]
    return x, lang_ids, positions


def add_bert_mask(x,
                  pad_index,
                  mask_index,
                  nwords,
                  p_pred=0.15,
                  p_mask=0.8,
                  p_keep=0.1,
                  p_rand=0.1,
                  fix=False):
    rng = np.random
    if fix: rng = np.random.RandomState(0)
    _x = x
    bsz, seqlen = x.size()
    pred_mask = torch.from_numpy(
        (rng.rand(bsz, seqlen) <= p_pred).astype(np.uint8))
    pad_mask = x == pad_index

    pred_mask[pad_mask] = 0
    x_real = x[pred_mask.bool()]

    x_rand = x_real.clone().random_(nwords)
    x_mask = x_real.clone().fill_(mask_index)
    probs = torch.multinomial(torch.tensor([p_keep, p_rand, p_mask]),
                              len(x_real),
                              replacement=True)
    fused_x = x_real * (probs == 0).long() + x_rand * (
        probs == 1).long() + x_mask * (probs == 2).long()
    x = x.masked_scatter(pred_mask.bool(), fused_x)

    # print("Origin:", _x.tolist())
    # print("Masked:", x.tolist())
    # print()
    return x, pred_mask, x_real


def load_pretrain_embedding(file, embedding, vocab):
    print("Loading Glove Model")
    f = open(file, 'r')
    gloveModel = {}
    for i, line in enumerate(f):
        splitLines = line.split()
        word = splitLines[0]
        try:
            wordEmbedding = np.array(
                [float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        except ValueError:
            pass

    for k, v in vocab.stoi.items():
        if k in gloveModel:
            embedding[v].copy_(torch.from_numpy(gloveModel[k]))
    embedding = nn.Embedding.from_pretrained(embedding)
    print("loaded")
    return embedding


def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    hits = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            hits += 1
    return hits / len(y_true)


def f1_recall_precision(y_true, y_pred):
    labels = list(set(y_true))
    assert len(y_true) == len(y_pred)
    res = {}
    for label in labels:
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == label:
                    tp += 1
                else:
                    tn += 1
            else:
                if y_true[i] == label:
                    fn += 1
                else:
                    fp += 1

        recall = tp / (tp + fn + 1e-10)
        precison = tp / (tp + fp + 1e-10)
        f1 = 2 * precison * recall / (recall + precison + 1e-10)
        res[label] = {'f1': f1, 'recall': recall, 'precison': precison}
    return res


def batch_data(data, pad_index, eos_index):
    lengths = [len(d) + 2 for d in data]
    tensor = torch.LongTensor(len(data), max(lengths)).fill_(pad_index)
    tensor[:, 0] = eos_index
    for i, s in enumerate(data):
        if lengths[i] > 2:
            tensor[i, 1:lengths[i] - 1].copy_(torch.LongTensor(data[i]))
        tensor[i, lengths[i] - 1] = eos_index
    return tensor


def truncate(data, max_seq_length, pad_index, eos_index):
    lengths = (data != pad_index).sum(dim=-1)
    if lengths.max() > max_seq_length:
        data = data[:, :max_seq_length].clone()
        for i in range(len(lengths)):
            if lengths[i] > max_seq_length and eos_index > 0:
                data[i, max_seq_length - 1] = eos_index
    return data


def to_cuda(*array):
    return [None if x is None else x.cuda() for x in array]


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.
    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new directory for the model.
    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(log_dir: str = None, mode: str = "train") -> None:
    """
    Create a logger for logging the training/testing process.
    :param log_dir: path to file where log is stored as well
    :param mode: log file name. 'train', 'test' or 'translate'
    :return: joeynmt version number
    """
    logger = logging.getLogger("")  # root logger
    #version = pkg_resources.require("joeynmt")[0].version

    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s  %(message)s')

        if log_dir is not None:
            if os.path.exists(log_dir):
                log_file = f'{log_dir}/{mode}.log'

                fh = logging.FileHandler(log_file, 'a')
                fh.setLevel(level=logging.DEBUG)
                logger.addHandler(fh)
                fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        logger.addHandler(sh)
        logger.info(
            "Welcome to TenTrans (Uniform Training & Decoding Platform) World !"
        )


def log_config(cfg: dict, prefix: str = "cfg") -> None:
    """
    Write configuration to log.
    :param cfg: configuration to log
    :param prefix: prefix for logging
    """
    logger = logging.getLogger(__name__)
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_config(v, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:50s} : {}".format(p, v))
            #print("{:34s} : {}".format(p, v))


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def bpe_postprocess(string, bpe_type="subword") -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.
    :param string:
    :param bpe_type: one of {"sentencepiece", "subword-nmt"}
    :return: post-processed string
    """
    if bpe_type == "sentencepiece":
        ret = string.replace(" ", "").replace("▁", " ").strip()
    elif bpe_type == "subword":
        ret = string.replace("@@ ", "").strip()
    else:
        ret = string.strip()
    return ret


def score_bleu(hypotheses, references, tokenize="13a"):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param tokenize: one of {'none', '13a', 'intl', 'zh', 'ja-mecab'}
    :return:
    """
    return sacrebleu.corpus_bleu(sys_stream=hypotheses,
                                 ref_streams=[references],
                                 tokenize=tokenize,
                                 force=True).score
