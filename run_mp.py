import argparse
import random
import os
import json
from copy import deepcopy
import glob
from pprint import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.utils import seed_everything
    

def run_exp(exp_name: str, exp_config: str, 
            run_type: str, nprocesses: int, opts=None) -> None:
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
    config.freeze()
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.system("mkdir -p data/logs/running_log")
    logger.add_filehandler('data/logs/running_log/' + config.LOG_FILE)
    logger.info(f"hyper parameters:\n{config.EVAL}")
    logger.info(f"llm reply file: {config.TASK_CONFIG.DATASET.LLM_REPLYS_PATH}")
    
    # dataset split, start multi-processes
    num_devices = torch.cuda.device_count()
    print(f'num devices: {num_devices}, num processes: {nprocesses}')
    with open(config.TASK_CONFIG.DATASET.LLM_REPLYS_PATH, 'r') as f:
        llm_reply_dataset = json.load(f)
    episode_ids = list(llm_reply_dataset.keys())
    split_episode_ids = [episode_ids[i::nprocesses] for i in range(nprocesses)]

    configs = []
    for i, ep_ids in enumerate(split_episode_ids):
        shared_config = deepcopy(config)
        shared_config.defrost()
        device_num = i % num_devices
        shared_config.local_rank = i
        shared_config.world_size = nprocesses
        shared_config.TORCH_GPU_ID = device_num
        shared_config.TORCH_GPU_IDS = [device_num]
        shared_config.SIMULATOR_GPU_IDS = [device_num]
        shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = ep_ids
        shared_config.freeze()
        configs.append(shared_config)
    
    pool = Pool(processes=nprocesses)
    pool.map(worker, configs)
    pool.close()
    pool.join()
    fns = glob.glob(config.CHECKPOINT_FOLDER + '/stats_ep_ckpt_*.json')
    summary = {}
    for fn in fns:
        with open(fn, 'r') as f:
            summary.update(json.load(f))
    summary_metrics = {
        "steps_taken": [],
        "distance_to_goal": [],
        "success": [],
        "oracle_success": [],
        "path_length": [],
        "spl": [],
        "ndtw": [],
        "sdtw": [],
    }
    for epid, metric in summary.items():
        for k, v in metric.items():
            summary_metrics[k].append(v)
    for k, v in summary_metrics.items():
        summary_metrics[k] = np.mean(v)
    pprint(summary_metrics)
    with open(config.CHECKPOINT_FOLDER + '/stats_ckpt_val_unseen.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)

def worker(config):
    seed_everything(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = TRAINER(config)
    trainer.eval()

if __name__ == "__main__":
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
        "--nprocesses",
        type=int,
        default=1,
        help="number of processes",
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
    
    mp.set_start_method('spawn', force=True)
    run_exp(**vars(args))