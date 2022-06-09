<!-- vscode-markdown-toc -->
<!-- * 1. [项目介绍](#)
* 2. [安装教程](#-1)
* 3. [快速上手预训练模型](#-1)
* 4. [更多任务](#-1)
* 5. [开源协议](#-1)
* 6. [联系方式](#-1) -->

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

[简体中文](readme_en.md)｜English

<p align="center">
  <img src="asserts/tentrans.png" width="200">
  </br>
  <a href="https://github.com/TenTrans/TenTrans/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-blue" /></a>
  <a href="https://github.com/TenTrans/TenTrans/releases"><img alt="Latest Release" src="https://img.shields.io/badge/Release-v1.0-yellow" /></a>
  <a href="https://github.com/TenTrans/TenTrans/releases"><img alt="Latest Release" src="https://img.shields.io/badge/Language-Python-orange" /></a>
</p>


-------
## 🔥 Introduction
TenTrans is a flexible and lightweight training framework for natural language processing, which supports various NLP tasks (including natural language understanding, generation and pre-training). It has the following advantages:
- Atomization and flexibility: Users can arbitrarily combine different NLP tasks for multi-task learning and the parameters between different tasks can be shared in a fine-grained manner.
- High-performance training: Tentrans supports training on multi nodes and multi gpus and successfully does pre-training on a 150G+ large corpus. 
- Multilingual and Cross-lingual： Users can employ the cross-lingual transfer learning methods to tackle the low-resource problems.
- Ecosystem：We provide efficent inference  engine [TenTrans-Decoding](https://github.com/TenTrans/TenTrans-Decoding).

Support Tasks
- Natural Language Generation（NMT）
- Natural Language Understanding（Classification, Cross-lingual NLU）
- Pretraining（MLM、TLM、MASS）

Support Features：
- Transformer
- Bert, XLM
- Distributed training(multi-node, multi-gpu)
- Classification(SST2, SST5)
- Fast beam search
- Average checkpoints and interactive inference



## ⚙️ Installation
```
git clone git@github.com:TenTrans/TenTrans.git
pip install -r requirements.txt 
```

## 🚀 Quick Start on Pre-training 
TenTrans supports various pre-training tasks, including encoder-based pretraining(e.g. MLM) and seq2seq-based pretraining(e.g. MASS). In addition, it also supports 
large-scale multilingual MT pretraining.

The following instruction will show how to train the pretraining model based MLM objective.

1. Data Process

We first should binarize the training data to acclerate the training process. The vocabulary file is in the form of one word per line. Then do the following command to generate the binary file.

```bash
python process.py vocab file  lang [shard_id](optional)
```


2. Configuration

TenTrans use the **yaml file**  for configuration. We provides several templates of different tasks(see **run/**/ directory). You can modify these files for adaptation.

```yaml
# base config
langs: [en]
epoch: 15
update_every_epoch:  1   
dumpdir: ./dumpdir       
share_all_task_model: True 
save_intereval: 1      
log_interval: 10       



optimizer: adam 
learning_rate: 0.0001
learning_rate_warmup: 4000
scheduling: warmupexponentialdecay
max_tokens: 2000
group_by_size: False   
max_seq_length: 260    
weight_decay: 0.01
eps: 0.000001
adam_betas: [0.9, 0.999]

sentenceRep:           
  type: transformer 
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


tasks:                
  en_mlm:             
    task_name: mlm    
    data:
        data_folder: your_data_folder
        src_vocab: vocab.txt
        train_valid_test: [train.bpe.en.pth, valid.bpe.en.pth, test.bpe.en.pth]
        stream_text: False  
        p_pred_mask_kepp_rand: [0.15, 0.8, 0.1, 0.1]

    target:           
        sentence_rep_dim: 768
        dropout: 0.1
        share_out_embedd: True
```
3. Training

Multi GPUS

```shell
export CUDA_VISIBLE_DEVICES=8;
python -m torch.distributed.launch \
                --nproc_per_node=$NPROC_PER_NODE main.py \
                --config run/xlm.yaml --multi_gpu True
```

##  📋 More Tasks
 - [Classification（SST2）](examples/TASK/SST2.md) 
 - [Machine Translation（WMT14ENDE）](examples/TASK/WMTENDE.md)

## 🔑 License
This project is released under MIT License.

## 🙋‍♂️ Contact information
For communication related to this project, please contact Baijun Ji(begosu@foxmail.com; baijunji@tencent.com) ，Bojie Hu(bojiehu@tencent.com)，Ambyera(ambyera@tencent.com）。




