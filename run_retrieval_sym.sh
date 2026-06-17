# ADD these two lines before the nohup command
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=1,2 nohup python -u \
    -m torch.distributed.launch \
    --nproc_per_node=2 --use_env --master_port=45577 \
    fashion_retrieval_sym.py \
    --config           configs/fashion_retrieval.yaml \
    --output_dir       output/finetune_retrieval5set_sym \
    --pre_point        output/full_run/epoch029/checkpoint.pth \
    --data_root        ./data/data-fashion \
    --catemap_filename fashion_annotation/categorys_to_sign.txt \
    --grad_accum_steps 4 \
    --subset_ratio     1.0 \
    > finetune_final.out 2>&1 &