
set -ex 
for tst in  nist02 ;do #
#tst=nist06
#for i in 13 14 15 16 17 18 19 20 21;do
i=17
CUDA_VISIBLE_DEVICES=1 /apdcephfs/share_1157259/users/baijunji/anaconda3/bin/python -u translation_infer.py \
        --src /apdcephfs/share_1157259/users/baijunji/data/train_data/ldc/$tst.bpe.in \
        --src_vocab /apdcephfs/share_1157259/users/baijunji/data/train_data/ldc/vocab.ch \
        --tgt_vocab /apdcephfs/share_1157259/users/baijunji/data/train_data/ldc/vocab.en \
        --src_lang ch \
        --tgt_lang en --batch_size 50 --beam 5 \
        --model_path $1 #> out.$tst
#/apdcephfs/share_1157259/users/baijunji/data/exp/tentrans/ldc_mt2/checkpoint_seq2seq_ldc_mt_
cat out.$tst |grep "Target_" | cut -f2- -d " " | sed -r 's/(@@ )|(@@ ?$)//g' >  predict.$tst
perl ../scripts/multi-bleu.perl -lc /apdcephfs/share_1157259/users/baijunji/data/train_data/ldc/$tst.ref.* < predict.$tst
done
