import os
import cv2
import numpy as np
import torch.nn as nn
from typing import Union, Tuple, List
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.constant import direction_mapping

from habitat import Config


class DirectionMap(nn.Module):
    def __init__(self, config: Config, full_map_shape: Union[Tuple, List, np.ndarray], 
                 theta: float=90.0, radius: float=5.0) -> None:
        super().__init__()
        self.config = config
        self.shape = full_map_shape
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
        self.resolution = config.MAP.MAP_RESOLUTION
        
        self.theta = theta
        self.radius = radius
        
        self.direction_map = np.zeros(self.shape)
    
    def reset(self):
        self.direction_map = np.ones(self.shape)
        
    def _create_sector_mask(self, position: np.ndarray, heading: float, direction_weight: float=1.5):
        """ 
        arg "position" came from full pose, full pose use standard Cartesian coordinate.
        """
        mask = np.zeros(self.shape)
        heading = (360 - heading) % 360
        angle_high = (heading + self.theta / 2) % 360
        angle_low = (heading - self.theta / 2) % 360
        heading = np.ones(self.shape) * heading

        y, x = np.meshgrid(np.arange(self.shape[0]) - position[0], np.arange(self.shape[1]) - position[1])
        distance = np.sqrt(x**2 + y**2)
        angle = np.arctan2(x, y) * 180 / np.pi
        angle = (360 - angle) % 360

        valid_distance = distance <= self.radius * 100 / self.resolution
        if angle_high > angle_low:
            valid_angle = (angle_low <= angle) & (angle <= angle_high)
        else:
            valid_angle = (angle_low <= angle) | (angle <= angle_high)
        mask[valid_distance & valid_angle] = direction_weight

        return mask
    
    def forward(self, current_position: np.ndarray, last_five_step_position: np.ndarray, heading: float, 
                direction: str, step: int, current_episode_id: int) -> np.ndarray:
        heading_vector = current_position - last_five_step_position
        if np.linalg.norm(heading_vector) <= 0.2 * 100 / self.resolution:
            heading_angle = heading
        else:
            heading_angle = angle_between_vectors(np.array([1, 0]), heading_vector)
        print("!!!!heading angle: ", heading_angle, direction, "left" in direction)
        direction = direction_mapping.get(direction, "ambiguous direction")
        if direction == "forward":
            sector_mask = self._create_sector_mask(current_position, heading_angle)
        elif direction == "left":
            heading_angle += 45
            sector_mask = self._create_sector_mask(current_position, heading_angle)
        elif direction == "right":
            heading_angle -= 45
            sector_mask = self._create_sector_mask(current_position, heading_angle)
        elif direction == "backward":
            heading_angle += 180
            sector_mask = self._create_sector_mask(current_position, heading_angle)
        else:
            sector_mask = np.ones(self.shape)
        
        if self.visualize:
            self._visualize(sector_mask, step, current_episode_id)
        
        return sector_mask
    
    def _visualize(self, direction_map: np.ndarray, step: int, current_episode_id: int):
        direction_map_vis = direction_map.copy()
        direction_map_vis[direction_map_vis == 0] = 1
        direction_map_vis = np.flipud((direction_map * 255).astype(np.uint8))
        cv2.imshow("history map", direction_map_vis)
        cv2.waitKey(1)
        
        if self.print_images:
            save_dir = os.path.join(self.config.RESULTS_DIR, "direction_map/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, direction_map_vis)