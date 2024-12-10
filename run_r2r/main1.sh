flag=" --exp_name exp_100trajs
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      NUM_ENVIRONMENTS 1
      "

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 12346 run.py $flag
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 run.py $flag