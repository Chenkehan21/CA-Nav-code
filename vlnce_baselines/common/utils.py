import os
import random
import numpy as np
import torch
import torch.distributed as distr

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    distr.all_gather(gather_t, tensor)
    
    return gather_t

def seed_everything(seed: int):
    # print(f"setting seed: {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_number):
    device = torch.device("cpu")
    if device_number >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{0}".format(device_number))

    return device