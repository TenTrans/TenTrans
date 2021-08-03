
set -ex 

CUDA_VISIBLE_DEVICES=0  /mnt///share_1399748/users/baijunji/anaconda3/bin/python -u translation_infer.py \
        --src /mnt///share_1399748/users/baijunji/data/train_data/wmt16_ende/test.bpe.en \
        --src_vocab /mnt///share_1399748/users/baijunji/data/train_data/wmt16_ende/vocab.bpe.32000 \
        --tgt_vocab /mnt///share_1399748/users/baijunji/data/train_data/wmt16_ende/vocab.bpe.32000 \
        --src_lang en \
        --tgt_lang de --batch_size 50 --beam 1 --length_penalty 0.6 --decode_by_length True \
       --model_path /mnt/share_1399748/users/baijunji/data/exp/tentrans/wmt14ende_base/checkpoint_seq2seq_wmtende_mt_$1  > out.ende
 
cat out.ende |grep "Target_" | cut -f2- -d " " | sed -r 's/(@@ )|(@@ ?$)//g' >  predict.ende
perl ../scripts/multi-bleu.perl  /mnt///share_1399748/users/baijunji/data/train_data/wmt16_ende/test.tok.de < predict.ende


cat  /mnt///share_1399748/users/baijunji/data/train_data/wmt16_ende/test.tok.de |  perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref
cat  predict.ende | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys

perl ../scripts/multi-bleu.perl generate.ref < generate.sys
