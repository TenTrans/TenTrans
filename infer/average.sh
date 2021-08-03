

path=/apdcephfs/share_1399748/users/baijunji/data/exp/tentrans/wmt14ende_8gpu_acc2/
 /apdcephfs/share_1399748/users/baijunji/anaconda3/bin/python  ../scripts/average_checkpoint.py --inputs  $path/checkpoint_seq2seq_ldc_mt_23 \
   $path/checkpoint_seq2seq_ldc_mt_24 $path/checkpoint_seq2seq_ldc_mt_25 $path/checkpoint_seq2seq_ldc_mt_26 $path/checkpoint_seq2seq_ldc_mt_27 \
   --output $path/checkpoint_seq2seq_ldc_mt_average


