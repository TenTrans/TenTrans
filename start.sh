 export NPROC_PER_NODE=8;

/apdcephfs/share_1399748/users//baijunji/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
      --master_port 2101 \
      main.py \
       --config run/mt_wmt14ende_big.yaml --multi_gpu True 


# /share_1399748/users//baijunji/anaconda3/bin/python main.py --config run/mt_ep.yaml
