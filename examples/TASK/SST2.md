### 文本分类（SST2）
您同样可以使用我们所提供的预训练模型来进行下游任务， 本节我们将以SST2任务为例， 让你快速上手使用预训练模型进行微调下游任务。
1. 数据处理

我们推荐使用文本格式进行文本分类的训练，因为这更轻量和快速。我们将SST2的数据处理为如下格式(见*sample_data* 文件夹):

| seq1 |lang1 | label1 | 
| :--- | :---: | ---: |
| This is a positive sentence. |en| postive | 
| This is a negtive sentence.| en|negtive | 
| This is a  sentence.| en |unknow | 

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
        feature: [seq1, lang1, label1]
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
