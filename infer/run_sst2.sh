set -ex
for label in 0.5 0.55 0.6  0.65 0.7  0.75 0.8 0.85 0.9 0.95;do


#  CUDA_VISIBLE_DEVICES=0 python -u classification_infer.py \
#         --model /apdcephfs/share_1157259/users/baijunji/data/exp/tentrans/eyi32w/eyi_1gpu_8bsz/checkpoint_classification_eyi_zhen_best  \
#         --vocab /apdcephfs/share_1157259/users/baijunji/data/train_data/eyi_process/eyi_xlm/eyi_32w//vocab_xnli_15.v2 \
#         --src /apdcephfs/share_1157259/users/baijunji/data/train_data/eyi_process/eyi_xlm/data/test.bpe.en.txt \
#         --lang en --threhold $label  > test.out.$label

# CUDA_VISIBLE_DEVICES=0 python -u classification_infer.py \
#         --model /data/home/baijunji/data/exp/tentrans/eyi_bsz8_1gpu2/checkpoint_classification_sst2_en_best2  \
#         --vocab /apdcephfs/share_1157259/users/baijunji/data/train_data/eyi_process/eyi_xlm/data/vocab_xnli_15 \
#         --src /apdcephfs/share_1157259/users/baijunji/data/train_data/eyi_process/eyi_xlm/data/test2.bpe.en.txt \
#         --lang en --threhold $label  > test2.out.$label
 echo $label
python ../scripts/eval_recall.py  /apdcephfs/share_1157259/users/baijunji/data/train_data/eyi_process/eyi_xlm/data//test.en.label test.out.$label
done