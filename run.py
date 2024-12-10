import argparse
import random
import os
import json

import numpy as np
import torch

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.config.default import get_config


def main():
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["eval"],
        required=True,
        help="run type of the experiment(train, eval, inference), only eval for zero-shot vln",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    print(args)
    run_exp(**vars(args), local_rank=local_rank)
    

def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()
    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE
    config.local_rank = local_rank
    with open(config.TASK_CONFIG.DATASET.LLM_REPLYS_PATH, 'r') as f:
        llm_reply_dataset = json.load(f)
    episode_ids = list(llm_reply_dataset.keys())
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episode_ids
    config.freeze()
    # import pdb;pdb.set_trace()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.system("mkdir -p data/logs/running_log")
    logger.add_filehandler('data/logs/running_log/' + config.LOG_FILE)
    
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.eval()
    

if __name__ == "__main__":
    main()