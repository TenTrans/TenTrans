- [项目介绍](#项目介绍)
- [安装教程](#安装教程)
- [快速上手](#快速上手)
  - [（一）预训练模型](#一预训练模型)
  - [（二）机器翻译](#二机器翻译)
  - [（三）文本分类](#三文本分类)



## 项目介绍
TenTrans是一个灵活轻量的自然语言处理训练框架， 支持常见的NLP任务（包括自然语言理解、生成、预训练）。 TenTrans有以下特点:

- 任务原子化： 用户可以任意组合各种NLP任务进行联合训练， 各任务之间的参数可以细粒度地共享。
- 高性能训练： TenTrans支持多机多卡大规模预训练，已在150G以上的超大规模语料上进行预训练验证。
- 多语言及跨语言： TenTrans支持跨语言预训练，用户可以通过迁移学习的方式解决低资源语言语料不足的问题。
- 多模态： TenTrans目前支持图片作为输入，包括OCR识别和图片翻译。

TenTrans目前支持的NLP任务包括
- 自然语言生成（机器翻译、图片翻译）
- 自然语言理解（文本分类,跨语言理解）
- 预训练（MLM、TLM、MASS）
- 多模态生成 （图片翻译）

支持特性：
- Transformer
- Bert, XLM
- Distributed training(multi-node, multi-gpu)
- Classification(SST2, SST5)
- Fast beam search
- Average checkpoints and interactive inference


## 安装教程
```
git clone git@github.com:TenTrans/TenTrans.git
pip install -r requirements.txt 
```
Tentrans是一个基于Pytorch的轻量级工具包，安装十分方便。

## 快速上手

### （一）预训练模型
TenTrans支持多种预训练模型，包括基于编码器的预训练（e.g. MLM）和基于seq2seq结构的生成式预训练方法（e.g. Mass）。 此外， Tentrans还支持大规模的多语言机器翻译预训练。

我们将从最简单的MLM预训练开始，让您快速熟悉TenTrans的运行逻辑。

1. 数据处理

在预训练MLM模型时，我们需要对单语训练文件进行二进制化。您可以使用以下命令, 词表的格式为一行一词，执行该命令后会生成train.bpe.en.pth。

```bash
python process.py vocab file  lang [shard_id](optional)
```

当数据规模不大时，您可以使用纯文本格式的csv作为训练文件。csv的文件格式为

| seq1 | lang1 |
| :--- |  ---: |
| This is a positive sentence. | en |
| This is a negtive sentence.| en |
| This is a  sentence.|  en |

2. 参数配置

TenTrans是通过yaml文件的方式读取训练参数的， 我们提供了一系列的适应各个任务的训练配置文件模版（见 **run/** 文件夹），您只要改动很小的一部分参数即可。

```yaml
# base config
langs: [en]
epoch: 15
update_every_epoch:  1   # 每轮更新多少step
dumpdir: ./dumpdir       # 模型及日志文件保存的地方
share_all_task_model: True # 是否共享所有任务的模型参数
save_intereval: 1      # 模型保存间隔
log_interval: 10       # 打印日志间隔



#全局设置开始， 如果tasks内没有定义特定的参数，则将使用全局设置
optimizer: adam 
learning_rate: 0.0001
learning_rate_warmup: 4000
scheduling: warmupexponentialdecay
max_tokens: 2000
group_by_size: False   # 是否对语料对长度排序
max_seq_length: 260    # 模型所能接受的最大句子长度
weight_decay: 0.01
eps: 0.000001
adam_betas: [0.9, 0.999]

sentenceRep:           # 模型编码器设置
  type: transformer #cbow, rnn
  hidden_size: 768
  ff_size: 3072
  dropout: 0.1
  attention_dropout: 0.1
  encoder_layers: 12
  num_lang: 1
  num_heads: 12
  use_langembed: False
  embedd_size: 768
  learned_pos: True
  pretrain_embedd: 
  activation: gelu
#全局设置结束


tasks:                #任务定义， TenTrans支持多种任务联合训练，包括分类，MLM和seq2seq联合训练。
  en_mlm:             #任务ID，  您可以随意定义有含义的标识名
    task_name: mlm    #任务名，  TenTrans会根据指定的任务名进行训练
    data:
        data_folder: your_data_folder
        src_vocab: vocab.txt
        # train_valid_test: [train.bpe.en.csv, valid.bpe.en.csv, test.bpe.en.csv]
        train_valid_test: [train.bpe.en.pth, valid.bpe.en.pth, test.bpe.en.pth]
        stream_text: False  # 是否启动文本流训练
        p_pred_mask_kepp_rand: [0.15, 0.8, 0.1, 0.1]

    target:           # 输出层定义
        sentence_rep_dim: 768
        dropout: 0.1
        share_out_embedd: True
```
3. 启动训练

单机多卡

```shell
export NPROC_PER_NODE=8;
python -m torch.distributed.launch \
                --nproc_per_node=$NPROC_PER_NODE main.py \
                --config run/xlm.yaml --multi_gpu True
```
### （二）机器翻译

本节您将快速学会如何训练一个基于Transformer的神经机器翻译模型，我们以WMT14 英-德为例（[下载数据](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8)）。

1. 数据处理

与处理单语训练文件相同，您也需要对翻译的平行语料进行二进制化。

```shell
python process.py vocab.bpe.32000 train.bpe.de de
python process.py vocab.bpe.32000 train.bpe.en en
```

2. 参数配置

```yaml
# base config
langs: [en, de]
epoch: 50
update_every_epoch: 5000
dumpdir: ./exp/tentrans/wmt14ende_template

share_all_task_model: True
optimizer: adam 
learning_rate: 0.0007
learning_rate_warmup: 4000
scheduling: warmupexponentialdecay
max_tokens: 16384  # src tokens + tgt tokens
max_seq_length: 512
save_intereval: 1
weight_decay: 0
adam_betas: [0.9, 0.98]

clip_grad_norm: 0
label_smoothing: 0.1
accumulate_gradients: 1
share_all_embedd: True
patience: 10
#share_out_embedd: False

tasks:
  wmtende_mt:
    task_name: seq2seq
    reload_checkpoint:
    data:
        data_folder:  /train_data/wmt16_ende/
        src_vocab: vocab.bpe.32000
        tgt_vocab: vocab.bpe.32000
        train_valid_test: [train.bpe.en.pth:train.bpe.de.pth, valid.bpe.en.pth:valid.bpe.de.pth, test.bpe.en.pth:test.bpe.de.pth]
        group_by_size: True
        max_len: 200

    sentenceRep:
      type: transformer 
      hidden_size: 512
      ff_size: 2048
      attention_dropout: 0.1
      encoder_layers: 6
      num_heads: 8
      embedd_size: 512
      dropout: 0.1
      learned_pos: True
      activation: relu

    target:
      type: transformer 
      hidden_size: 512
      ff_size: 2048
      attention_dropout: 0.1
      decoder_layers: 6
      num_heads: 8
      embedd_size: 512
      dropout: 0.1
      learned_pos: True
      activation: relu
```

3. 模型解码

大约训练更新10万步之后（8张V100，大约10小时）， 我们可以使用TenTrans提供的脚本对平均最后几个模型来获得更好的效果。
```shell
path=model_save_path
python  scripts/average_checkpoint.py --inputs  $path/checkpoint_seq2seq_ldc_mt_18 \
    $path/checkpoint_seq2seq_ldc_mt_19 $path/checkpoint_seq2seq_ldc_mt_20 \
    $path/checkpoint_seq2seq_ldc_mt_21 $path/checkpoint_seq2seq_ldc_mt_22 \
    $path/checkpoint_seq2seq_ldc_mt_23  \
    --output $path/average.pt
```
我们可以使用平均之后的模型进行翻译解码，
```shell
python -u infer/translation_infer.py \
        --src train_data/wmt16_ende/test.bpe.en \
        --src_vocab train_data/wmt16_ende/vocab.bpe.32000 \
        --tgt_vocab train_data/wmt16_ende/vocab.bpe.32000 \
        --src_lang en \
        --tgt_lang de --batch_size 50 --beam 4 --length_penalty 0.6 \
        --model_path model_save_path/average.pt | \
        grep "Target_" | cut -f2- -d " " | sed -r 's/(@@ )|(@@ ?$)//g' > predict.ende

cat  train_data/wmt16_ende/test.tok.de |  perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref
cat  predict.ende | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
perl ../scripts/multi-bleu.perl generate.ref < generate.sys
```
4. 翻译结果

| WMT14-ende | BLEU | 
| ------ | ------ | 
| Attention is all you need(beam=4) | 27.30 | 
| TenTrans(beam=4, 8gpus, updates=200k, gradient_accu=1) | 27.54 | 
| TenTrans(beam=4, 8gpus, updates=125k, gradient_accu=2) | 27.64 | 
| TenTrans(beam=4, 24gpus, updates=90k, gradient_accu=1) | 27.67 |  

### （三）文本分类

您同样可以使用我们所提供的预训练模型来进行下游任务， 本节我们将以SST2任务为例， 让你快速上手使用预训练模型进行微调下游任务。

1. 数据处理

我们推荐使用文本格式进行文本分类的训练，因为这更轻量和快速。我们将SST2的数据处理为如下格式(见*sample_data* 文件夹):

| seq1 | label1 | lang1 |
| :--- | :---: | ---: |
| This is a positive sentence. | postive | en |
| This is a negtive sentence.| negtive | en |
| This is a  sentence.| unknow | en |

2. 参数配置

```yaml
# base config
langs: [en]
epoch: 200
update_every_epoch: 1000
share_all_task_model: False
batch_size: 8 
save_interval: 20
dumpdir: ./dumpdir/sst2

sentenceRep:
  type: transformer
  pretrain_rep: ../tentrans_pretrain/model_mlm2048.tt

tasks:
  sst2_en:
    task_name: classification
    data:
        data_folder:  sample_data/sst2
        src_vocab: vocab_en
        train_valid_test: [train.csv, dev.csv, test.csv]
        label1: [0, 1]
        feature: [seq1, label1, lang1]
    lr_e: 0.000005  # encoder学习率
    lr_p: 0.000125  # target 学习率
    target:
      sentence_rep_dim: 2048
      dropout: 0.1
    weight_training: False # 是否采用数据平衡
```
3. 分类解码
```shell
python -u classification_infer.py \
         --model model_path \
         --vocab  sample_data/sst2/vocab_en \
         --src test.txt \
         --lang en --threhold 0.5  > predict.out.label
python scripts/eval_recall.py  test.en.label predict.out.label
```





