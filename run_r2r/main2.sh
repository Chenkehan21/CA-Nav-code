export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag=" --exp_name exp_mp_1839_max_constraint_value_threshold0.3
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      --nprocesses 16
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0,1,2,3,4,5,6,7]
      SIMULATOR_GPU_IDS [0,1,2,3,4,5,6,7]
      "
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_mp.py $flag

# flag=" --exp_name exp_1_choice_check
#       --run-type eval
#       --exp-config vlnce_baselines/config/exp1.yaml
#       --nprocesses 1
#       NUM_ENVIRONMENTS 1
#       TRAINER_NAME ZS-Evaluator-mp
#       TORCH_GPU_IDS [0]
#       SIMULATOR_GPU_IDS [0]
#       "
# CUDA_VISIBLE_DEVICES=2 python run_mp.py $flag