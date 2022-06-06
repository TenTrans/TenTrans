###  机器翻译（WMT14ENDE）

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
| TenTrans(beam=4, 8gpus, updates=100k, gradient_accu=1) | 27.74 | 
