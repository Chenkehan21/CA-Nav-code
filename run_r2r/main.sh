flag=" --exp_name exp_debug
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      NUM_ENVIRONMENTS 1
      KEYBOARD_CONTROL 0
      TRAINER_NAME ZS-Evaluator-mp-multi_value_map
      "

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 12346 run.py $flag
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 run.py $flag