# 一、CXMT预训练语言模型
## 1. 预训练语言模型
### 1.1 说明
为鼓励更多参赛者加入比赛，减小参赛者预训练语言模型的计算代价，举办方发放基于Tentrans平台的预训练语言模型（Tiny, Base, Large）。
### 1.2 下载方式
|模型    | 链接  | 备注  |
| :------------ | :------------ | :------------ |
| Tiny   | [tiny.pretrain_model.zip](https://share.weiyun.com/ocTD0orN) | 使用Tentrans平台自训练 |
| Base  | [base.pretrain_model.zip](https://share.weiyun.com/aUXPCKt2) | 使用Tentrans平台自训练 |
| Large  | [large.pretrain_model.zip](https://share.weiyun.com/wnWkLgt7) | 为节省计算资源，我们直接使用tentrans平台， 复用[XLMR](https://arxiv.org/abs/1911.02116) 模型的参数，针对XLMR-LARGE不支持某些小语种的问题，我们在XLMR-LARGE的词表基础上，额外追加了一部分小语种特有token（新词表为xlmr.vocab.add），并将这些token对应的embedding做随机初始化，然后继续做continue pretrain。 |

### 1.3 模型参数
|模型    | hidden_size  | ff_size  | encoder_layers | embedd_size | num_heads |
| :------------ | :------------ | :------------ |:------------ |:------------ |:------------ |
| Tiny   | 128| 512 | 2 | 128 | 2 |
| Base  | 768 | 3072| 12| 768| 12 | 
| Large  | 1024 | 4096 | 24 | 1024 | 16 |

# 二、BASELINE
## 1. XTC
说明：以下结果使用Tentrans平台代码进行finetune, 在各个语种的Dev上选择weighted-F1得分最高的模型，然后测试该模型在各个语种的TestA上的得分

### 1.1 Tiny
|Tentrans-Tiny  | zh |	mn	| bo|	ug	|ct	| en	| kk	| ko	| average 
| :------------ | :------------ | :------------ |:------------ |:------------ |:------------ |:------------ |:------------ |:------------ |:------------ |
| Dev |70.1	|20.6	|20.4	|17.5	|61	|25	|19	|28.8	|32.8|
| TestA |70.4	|22.8	|20.4	|17.1	|62.2	|23.5	|18.7	|29.9	|33.1|

### 1.2 Base
|Tentrans-Base  | zh |	mn	| bo|	ug	|ct	| en	| kk	| ko	| average  |
| :------------ | :------------ | :------------ |:------------ |:------------ |:------------ |:------------ |:------------ |:------------ |:------------ |
| Dev | 73.1	|65	|45.9	|60.3	|69.4	|67.1	|52.4	|66.3| 62.4 |
| TestA | 72.4|66|43.9|60.4|69.9|69.5|51.8|64.2	|62.3|

### 1.3 Large
|Tentrans-Large  | zh |	mn	| bo|	ug	|ct	| en	| kk	| ko	| average  |
| :------------ | :------------ | :------------ |:------------ |:------------ |:------------ |:------------ |:------------ |:------------ |:------------ |
| Dev | 72.8 | 64.9|	48.8|	61.7| 70.6|	68.8| 53	|67.5 | 63.5|
| TestA |72.8|	65.5|	47.8|	62.2|	70.8|	71.4|	51.6|	65.1| 63.4 |

## 2. XSTS

# 三、BASELINE复现方式
## 1. XTC
### 1.1 Tiny&Base
下载代码：[Teg-Tentrans-xtc-tiny_base-finetune.zip](https://share.weiyun.com/A1NwpeSz)
#### 1.1.1 Finetune
以Tiny为例
```
lang=bo
lr_p=0.000005 # lr_p in [0.00000125, 0.0000025, 0.000005]
lr_e=0.000005 # lr_e in [0.00000125, 0.0000025, 0.000005]
batch_size=32 # batch in [8,16,32]

cd Teg-Tentrans-xtc-tiny_base-finetune

python3 main.py \
--config run/${lang}_xtc_tiny.yaml \
--lr_p ${lr_p} \
--lr_e ${lr_e} \
--batch_size ${batch_size} \
--dumpdir bz_${batch_size}+lr_p.${lr_p}+lr_e.${lr_e} \
--data_folder xtc_tiny-base/${lang} \
--pretrain_rep tiny.pretrain_model/checkpoint_mlm_en_mlm_700
```
#### 1.1.2 Infer
以Tiny为例
```
cd Teg-Tentrans-xtc-tiny_base-finetune/infer
sh run_infer.sh ${input} ${data_path} ${model_path}
输入说明：
input: 经过预处理（sentence piece）的小语种输入文本
data_path: 词表所在的文件夹路径
model_path: finetune产生的模型
```

### 1.2 Large
下载代码[Teg-Tentrans-xtc-large-finetune.zip](https://share.weiyun.com/MgpyAksP)（为了兼容XLMR模型，代码在TenTrans的主分支基础上有所修改）
#### 1.2.1 Finetune
```
lang=bo
lr_p=0.000005 # lr_p in [0.00000125, 0.0000025, 0.000005]
lr_e=0.000005 # lr_e in [0.00000125, 0.0000025, 0.000005]
batch_size=32 # batch in [8,16,32]

cd Teg-Tentrans-xtc-large-finetune

python3 main.py \
--config run/${lang}_xtc_large.yaml \
--lr_p ${lr_p} \
--lr_e ${lr_e} \
--batch_size ${batch_size} \
--dumpdir bz_${batch_size}+lr_p.${lr_p}+lr_e.${lr_e} \
--data_folder xtc_large/${lang} \
--pretrain_rep large.pretrain_model/checkpoint_mlm_mlm_650
```
#### 1.2.2 Infer
```
cd Teg-Tentrans-xtc-large-finetune/infer
sh run_infer.sh ${input} ${data_path} ${model_path}
输入说明：
input: 经过预处理（sentence piece）的小语种输入文本
data_path: 词表所在的文件夹路径
model_path: finetune产生的模型

```

## 2. XSTS
### 2.1 Tentrans
|  |	en	| bo|	ct	|ug	| mn	| ko	| kk
| :------------ | :------------ | :------------ |:------------ |:------------ |:------------ |:------------ |:------------ |
| Tentrans-Tiny | 1.7	|17.5	|67.0	|14.0	|19.0	|1.6	|15.7	|
| Tentrans-Base | 47.1	|67.8	|74.4	|69.1	|64.4	|21.6	|73.2	|



