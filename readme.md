- [项目介绍](#项目介绍)
- [安装教程](#安装教程)
- [快速上手](#快速上手)
  - [（一）预训练模型](#一预训练模型)
  - [（二）机器翻译](#二机器翻译)
  - [（三）文本分类](#三文本分类)
- [常见问题  FAQ](#常见问题--faq)
- [行为准则](#行为准则)
- [如何加入](#如何加入)
- [团队介绍](#团队介绍)
- [问题反馈](#问题反馈)


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
git clone git@git.woa.com:baijunji/Teg-Tentrans.git
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
max_tokens: 8000
max_seq_length: 512
save_intereval: 1
weight_decay: 0
adam_betas: [0.9, 0.98]

clip_grad_norm: 0
label_smoothing: 0.1
accumulate_gradients: 2
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

大约训练更新20万步之后（8张M40，大约耗时四十小时）， 我们可以使用TenTrans提供的脚本对平均最后几个模型来获得更好的效果。
```shell
path=model_save_path
python  scripts/average_checkpoint.py --inputs  $path/checkpoint_seq2seq_ldc_mt_40 \
    $path/checkpoint_seq2seq_ldc_mt_39 $path/checkpoint_seq2seq_ldc_mt_38 \
    $path/checkpoint_seq2seq_ldc_mt_37 $path/checkpoint_seq2seq_ldc_mt_36 \
    $path/checkpoint_seq2seq_ldc_mt_35 $path/checkpoint_seq2seq_ldc_mt_34 \
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

## 常见问题  FAQ

- Q: TenTrans是否支持太极训练平台？
  
  A: 支持。此外还支持太极的多机多卡预训练，具体配置文件见start.sh

- Q: TenTrans与Fairseq、OpenNMT等框架的主要区别是？
  
  A: Fairseq的定位是序列生成框架，OpenNMT只关注于机器翻译。 TenTrans并没有限定在特定的某个NLP任务上，用户可以很灵活地自定义自己的任务，无论是NLU、NLG或者是Pre-Training。目前TenTrans在NLG和Pre-Training的功能较为完善， NLU相对薄弱。

- Q: 训练速度和模型性能和Fairseq比怎么样？
  
  A: 速度和性能和Fairseq相当

- Q: 是否有配套的C++解码器？

  A: TenTrans配套有高性能的解码器，请移步我们的另一个解码器项目（https://git.woa.com/TenTrans/TenTrans-Decoding）。

- Q: Tentrans是否支持subword, spm等多种子词压缩技术？

  A: 支持，并且在训练过程中可自定义配置。

## 行为准则

TencentMT Oteam PMC、所有成员在oteam推进过程中需要遵守的行为准则简要列举如下：
1. TencentMT Oteam是公司级别的翻译开源协同团队，欢迎全公司的同学参与其中。
2. TencentMT Oteam的代码库：https://git.woa.com/TencentMT/TencentNMT, 最新的需求使用该项目。
3. TencentMT Oteam内部开源的代码、文档、数据等，不能私自转播到外部平台等。
4. TencentMT Oteam建议采用issue方式记录进展、沟通问题等。
5. TencentMT Oteam建议在提交代码、文档、数据的同时发布CR/MR到所有成员，approve之后再merge到主master分支。
6. TencentMT Oteam成员需按时参加定期组织的会议等。
7. Oteam相关页面：
    技术图谱：https://techmap.woa.com/oteam/8692
    iwiki：https://iwiki.woa.com/space/TencentMT
8. 代码更新规范：
    开发首先根据master建立自己的dev branch(如poetniu_dev)
    开发特性需同步新开issue
    提交模块代码（包括文档等）到dev branch，之后发CR/MR到所有PMC成员，统一通过之后merge到master branch
9. 其他补充：

    Our Pledge
    We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size,
visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality,
personal appearance, race, religion, or sexual identity and orientation.

    We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

    Our Standards
        Examples of behavior that contributes to a positive environment for our community include:

        Demonstrating empathy and kindness toward other people
        Being respectful of differing opinions, viewpoints, and experiences
        Giving and gracefully accepting constructive feedback
        Accepting responsibility and apologizing to those affected by our mistakes, and learning from the experience
        Focusing on what is best not just for us as individuals, but for the overall community
        
        Examples of unacceptable behavior include:

        The use of sexualized language or imagery, and sexual attention or advances of any kind
        Trolling, insulting or derogatory comments, and personal or political attacks
        Public or private harassment
        Publishing others' private information, such as a physical or email address, without their explicit permission
        Other conduct which could reasonably be considered inappropriate in a professional setting


## 如何加入
如何加入TencentMT Oteam？

说明

    加入Oteam，为Oteam贡献力量，主要有以下几个方向：
    贡献优化代码
        TNMT框架：https://git.woa.com/TencentMT/TencentNMT
        包括如下子模块：
            业务接口
            前处理：语种识别、tokenizer等
            通用翻译引擎：基于transformer的nmt引擎、引擎加速、模型蒸馏等
            后处理
            交互翻译引擎：翻译记忆融合、限制解码过程
        注意：贡献的代码必须codecc检查没有问题（100分），没有代码规范、代码安全、代码度量方面的问题。

        代码安全两个保证：

            提交前，codecc检查通过（100分）
            提交（merge后），立即运行codecc检查，如果发现有问题，必须当天内解决
        代码质量可以参考：代码质量

    贡献开源数据
        开源数据目前存放在企业微盘，可以找poetniu增加权限

    贡献文档
        补充文档：https://git.woa.com/TencentMT/TencentNMT
        提交issue

    TD

关注oteam

    关注oteam：https://techmap.woa.com/oteam/8692
    代码：https://git.woa.com/TencentMT/TencentNMT
    码客圈子：https://mk.woa.com/coterie?source=851，可以发布提问等

准备材料

    团队加入
        准备基础材料：
        目前业务方向，翻译应用场景，问题挑战等
        加入oteam的需求
        已有的翻译技术储备

    个人加入
        关注oteam
        贡献代码和数据
        增加相应权限


## 团队介绍 ##

PMC（Project Management Committees，项目管理委员会) 成员

    WXG-搜索应用部：poetniu，fandongmeng，withtomzhou
    CSIG-智能平台产品部：ethanlli
    TEG-信息安全部：bojiehu，springhuang、damonju
    TEG-AI Lab：donkeyhuang、shumingshi


各协同团队中涉及的产品或职能分工
    
    PMC成员&协同团队人员

    BG/部门 接口人  参与人员

    WXG-搜索应用部

        poetniu

        fandongmeng、withtomzhou

    CSIG-智能平台产品部

        frostwu

        ethanlli

    TEG-AI Lab

        donkeyhuang

        zptu、shumingshi

    TEG-信息安全部

        bojiehu

        ambyera、bengiojiang、damonju


各模块组件详细分工

模块    协调方  主要工作    重点协同目标组件    共建方

业务应用

    CSIG-智能平台部

    统一接口协议

    业务方接口协议、系统组件接口协议

    CSIG-智能平台部

前处理

    TEG-信息安全部

    开源、构建通用的前处理组件

    语种识别、干预策略

    WXG-搜索应用部、TEG-信息安全部

    文本预处理（中、英）

    WXG-搜索应用部

    文本预处理（日、韩）

    CSIG-智能平台部

    文本预处理（民族语言，如蒙、藏、维、粤等）

    TEG-信息安全部

    多媒体识别&特征提取（ASR、OCR、图像）

    TEG-信息安全部

通用翻译引擎

    WXG-搜索应用部

    开源、构建通用翻译引擎，优化组件

    神经翻译引擎（Transformer及变种）

    WXG-搜索应用部、TEG-信息安全部、CSIG-智能平台部

    多模态翻译引擎

    TEG-信息安全部

    模型压缩、推理加速

    WXG-搜索应用部、TEG-信息安全部、CSIG-智能平台部

交互翻译引擎

    TEG-AI Lab

    开源、构建交互翻译引擎，优化组件

    辅助翻译输入法

    TEG-AI Lab

    约束解码

    TEG-AI Lab

    翻译记忆融合

    TEG-AI Lab

后处理

    CSIG-智能平台部

    开源、构建后处理组件

    翻译检查

    CSIG-智能平台部、WXG-搜索应用部

    译文纠错

    CSIG-智能平台部、WXG-搜索应用部

离线模块

    TEG-AI Lab

    开源、构建离线工具

    工具类：词对齐、短语对抽取、数据清洗组件

    TEG-AI Lab

## 问题反馈
如果在使用过程中有任何问题或建议，或者遇到badcase，推荐使用GIT上的Issue功能提交反馈。




