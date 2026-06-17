#!/bin/bash

# Prevent NCCL P2P and IB clustering issues in multi-GPU setups
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=1,3 nohup python -u \
    -m torch.distributed.launch \
    --nproc_per_node=2 --use_env --master_port=45577 \
    fashion_catereg_sym.py \
    --config           configs/fashion_catereg.yaml \
    --output_dir       output/finetune_categoryhighLR_sym \
    --pre_point        /aul/homes/hsale014/Project1/Fashion/FashionSAP/output/full_run/epoch029/checkpoint.pth \
    --data_root        ./data/data-fashion \
    --task             both \
    --grad_accum_steps 4 \
    --subset_ratio     1.0 \
    > finetune_catereg_highLR.out 2>&1 &