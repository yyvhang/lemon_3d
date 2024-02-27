CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --save_checkpoint_path runs/LEMON/ \
    --batch_size 12 --yaml config/train.yaml