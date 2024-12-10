import glob
import json
import numpy as np
from pprint import pprint

fns = glob.glob('data/checkpoints/exp_100trajs' + '/stats_ep_ckpt_*.json')
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
with open('tmp.json', 'w') as f:
    json.dump(summary_metrics, f)