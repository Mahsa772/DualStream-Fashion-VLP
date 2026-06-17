NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=46009 fashion_tgir.py --data_root /a/bear.cs.fiu.edu./disk/bear-b/users/hsale014/Project1/Fashion/FashionSAP/data/data-fashion/FashionIQ --output_dir ./output_tgir/ >show_tgir.out 2>&1 &


#NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port=46006 fashion_tgir.py --data_root /a/bear.cs.fiu.edu./disk/bear-b/users/hsale014/Project1/Fashion/FashionSAP/data/data-fashion/FashionIQ --output_dir ./output_tgir/ >show_tgir.out 2>&1 &

#CUDA_VISIBLE_DEVICES=1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=3 --use_env --master_port=46005 fashion_tgir.py --data_root /a/bear.cs.fiu.edu./disk/bear-b/users/hsale014/Project1/Fashion/FashionSAP/data/data-fashion/FashionIQ --output_dir ./output_tgir/ >show_tgir.out 2>&1 &

#CUDA_VISIBLE_DEVICES=1 python -u fashion_tgir.py --data_root ./data-fashion/FashionIQ --output_dir ./output_tgir/

#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=46000 fashion_tgir.py --data_root ./data-fashion/FashionIQ --output_dir ./output_tgir/ >show_tgir.out 2>&1 &

#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=45999 fashion_tgir.py --data_root ./data-fashion/FashionIQ --output_dir ./output_tgir/ >show_tgir.out 2>&1 &

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=48888 fashion_tgir.py --data_root ./data-fashion/FashionIQ --test_only > show_eval_tgir.out 2>&1

#CUDA_VISIBLE_DEVICES=1 nohup python -u -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=45999 fashion_tgir.py --checkpoint my_retrieval_model.pth --data_root ./data-fashion/FashionIQ --output_dir ./output_tgir/ >show_tgir.out 2>&1 &

#CUDA_VISIBLE_DEVICES=1 nohup python -u -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=45999 fashion_tgir.py --checkpoint checkpoint_best.pth --data_root ./data/data-fashion --output_dir ./output_tgir/ >show_tgir.out 2>&1 &


#CUDA_VISIBLE_DEVICES=0,1 nohup python -u -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=45999 fashion_tgir.py >show_tgir.out 2>&1 &
