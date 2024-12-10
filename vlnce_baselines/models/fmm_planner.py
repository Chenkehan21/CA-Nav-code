import cv2
import copy
import skfmm
import numpy as np
from numpy import ma
from typing import Tuple

from vlnce_baselines.utils.map_utils import get_mask, get_dist


class FMMPlanner:
    def __init__(self, config, traversible: np.ndarray, scale: int=1, step_size: int=5, visualize: bool=False) -> None:
        self.scale = scale
        self.step_size = step_size
        self.visualize = visualize
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.)) # du=5
        self.fmm_dist = None
        self.waypoint_threshold = config.EVAL.FMM_WAYPOINT_THRESHOLD
        self.goal_threshold = config.EVAL.FMM_GOAL_THRESHOLD
        self.resolution = config.MAP.MAP_RESOLUTION
        
    def set_goal(self, goal: np.ndarray) -> None:
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = goal

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
    
    def get_short_term_goal(self, agent_position: np.ndarray, fixed_destination: np.ndarray, pad: int=5, ) -> Tuple:
        # dist = copy.deepcopy(self.fmm_dist)
        dist = np.pad(self.fmm_dist, pad, 'constant', constant_values=np.max(self.fmm_dist))
        # print(dist.sum())
        # if abs(dist.sum() - 44442594.984901555) < 1:
        #     import pdb;pdb.set_trace()
        # if self.visualize:
        #     normalized_data = ((dist - np.min(dist)) / 
        #                     (np.max(dist) - np.min(dist)) * 255).astype(np.uint8)
        #     normalized_data = np.stack((normalized_data,) * 3, axis=-1)
        #     cv2.imshow("padded dist", np.flipud(normalized_data))
        #     cv2.waitKey(1)
        x, y = int(agent_position[0]), int(agent_position[1])
        dx, dy = agent_position[0] - x, agent_position[1] - y
        mask = get_mask(dx, dy, scale=1, step_size=5)
        dist_mask = get_dist(dx, dy, scale=1, step_size=5)
        x += pad
        y += pad
        subset = dist[x - 5 : x + 6, y - 5: y + 6].copy()
        if subset.shape != mask.shape:
            print("subset and mask have different shape")
            print(f"subset shape:{subset.shape}, mask shape:{mask.shape}")
            print(f"current positon:{agent_position}")
            return x, y, True
        subset *= mask
        subset += (1 - mask) * 1e5
        if subset[5, 5] < self.waypoint_threshold * 100 / self.resolution:
            stop = True
        if fixed_destination is not None and subset[5, 5] < self.goal_threshold * 100 / self.resolution:
            stop = True
        else:
            stop = False
        # subset -= subset[5, 5]
        # ratio = subset / dist_mask
        # subset[ratio < -1.5] = 1
        
        # if self.visualize:
        #     subset_vis = 255 * (subset - np.min(subset)) / (np.max(subset) - np.min(subset))
        #     valid_subset = subset_vis[subset_vis != 255]
        #     valid_subset = (255 * (valid_subset - np.min(valid_subset)) / (np.max(valid_subset) - np.min(valid_subset)))
        #     subset_vis[subset_vis != 255] = valid_subset
        #     cv2.imshow("subset", np.flipud(subset_vis.astype(np.uint8)))
        #     cv2.imshow("subset2", np.flipud(dist[x - 5 : x + 6, y - 5: y + 6].astype(np.uint8)))
        #     cv2.waitKey(1)
        
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)
        offset_x, offset_y = stg_x - 5, stg_y - 5
        goal_x = x + offset_x - pad
        goal_y = y + offset_y - pad
        # print("stgx, stgy", stg_x, stg_y)
        # print("agent position: ", agent_position)
        # print("goal position: ", goal_x, goal_y)
        
        return goal_x, goal_y, stop