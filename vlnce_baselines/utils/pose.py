import numpy as np
from typing import List, Tuple
from collections import Sequence


def get_agent_position(agent_pose: Sequence, resolution: float) -> Tuple[np.ndarray, float]:
    x, y, heading = agent_pose
    heading *= -1
    x, y = x * (100 / resolution), y * (100 / resolution)
    position = np.array([y, x])
    
    return position, heading


def threshold_poses(coords: List, shape: Sequence) -> List:
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    
    return coords