set -ex

data_path=//share_1399748/users/baijunji/data/train_data/sst2.v2/xlm_inner

 CUDA_VISIBLE_DEVICES=0 //share_1399748/users//baijunji/anaconda3/bin/python  -u classification_infer.py \
        --model //share_1399748/users/baijunji/data/exp/tentrans/sst2_debug/checkpoint_classification_sst2_en_best  \
        --vocab $data_path/tentrans.vocab \
        --src   $data_path/dev.bpe.txt\
        --lang en   > test.out

//share_1399748/users//baijunji/anaconda3/bin/python  ../scripts/eval_recall.py  $data_path/dev.label test.out