import os
import cv2
import numpy as np
import torch.nn as nn
from typing import List
from fast_slic import Slic
from collections import Sequence
from scipy.spatial.distance import cdist
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.models.superpixel_waypoint_selector import WaypointSelector

import time
from pyinstrument import Profiler


class SuperPixelPolicy(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
        
        self.waypoint_selector = WaypointSelector(config)
    
    def reset(self) -> None:
        self.waypoint_selector.reset()
    
    def _get_sorted_regions(self, full_map: np.ndarray, traversible: np.ndarray, value_map: np.ndarray, 
                            collision_map: np.ndarray, detected_classes: OrderedSet) -> List:
        # floor = process_floor(full_map, detected_classes)
        # traversible = get_traversible_area(full_map, detected_classes)
        # t1 = time.time()
        valid_mask = value_map.astype(bool)
        min_val = np.min(value_map)
        max_val = np.max(value_map)
        normalized_values = (value_map - min_val) / (max_val - min_val + 1e-5)
        normalized_values[value_map == 0] = 1
        img = cv2.applyColorMap((normalized_values * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20.0) 
        slic.iterate(10)
        mask_slic = slic.getLabelContourMask()
        mask_slic *= valid_mask
        label_slic = slic.getLabels()
        label_slic *= valid_mask
        valid_labels = np.unique(label_slic)[1:]
        value_regions = []
        for label in valid_labels:
            mask = np.zeros_like(mask_slic)
            mask[label_slic == label] = 1
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if np.sum(output == 1) < 100:
                continue
            waypoint = np.array([int(centroids[1][1]), int(centroids[1][0])])
            value_mask = np.zeros_like(value_map)
            value_mask[waypoint[0] - 5: waypoint[0] + 5, waypoint[1] - 5: waypoint[1] + 5] = 1
            masked_value = value_mask * value_map
            waypoint = get_nearest_nonzero_waypoint(traversible, waypoint)
            if np.sum(collision_map) > 0:
                nonzero_indices = np.argwhere(collision_map != 0)
                distances = cdist([waypoint], nonzero_indices)
                if np.min(distances) <= 5:
                    print("!!!!!!!!!!!!!!!!!waypoint close to collision area, change waypoint!")
                    continue
            value_regions.append((mask, np.mean(masked_value[masked_value != 0]), waypoint))
        sorted_regions = sorted(value_regions, key=lambda x: x[1], reverse=True)
        waypoint_values =  np.array([item[1] for item in value_regions])
        # t2 = time.time()
        # print("real super pixel time: ", t2 - t1)
        return sorted_regions

    def _get_sorted_region_fast_slic(self, full_map: np.ndarray, traversible: np.ndarray, value_map: np.ndarray, 
                            collision_map: np.ndarray, detected_classes: OrderedSet) -> List:
        valid_mask = value_map.astype(bool)
        min_val = np.min(value_map)
        max_val = np.max(value_map)
        normalized_values = (value_map - min_val) / (max_val - min_val + 1e-5)
        normalized_values[value_map == 0] = 1
        img = cv2.applyColorMap((normalized_values * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        slic = Slic(num_components=24**2, compactness=100)
        assignment = slic.iterate(img)
        assignment *= valid_mask
        valid_labels = np.unique(assignment)[1:]
        value_regions = []
        # t1 = time.time()
        for label in valid_labels:
            mask = np.zeros_like(value_map)
            mask[assignment == label] = 1
            nb_components, output, stats, centroids = \
                cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            if np.sum(output == 1) < 100:
                continue
            waypoint = np.array([int(centroids[1][1]), int(centroids[1][0])])
            # if np.sum(collision_map) > 0:
            #     nonzero_indices = np.argwhere(collision_map != 0)
            #     distances = cdist([waypoint], nonzero_indices)
            #     if np.min(distances) <= 5:
            #         print("!!!!!!!!!!!!!!!!!waypoint close to collision area, change waypoint!")
            #         continue
            # waypoint = get_nearest_nonzero_waypoint(traversible, waypoint)
            value_mask = np.zeros_like(value_map)
            value_mask[waypoint[0] - 5: waypoint[0] + 5, waypoint[1] - 5: waypoint[1] + 5] = 1
            masked_value = value_mask * value_map
            # import pdb;pdb.set_trace()
            if np.sum(masked_value) > 0:
                value_regions.append((mask, np.mean(masked_value[masked_value != 0]), waypoint))
            else:
                value_regions.append((mask, 0., waypoint))
        # t2 = time.time()
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! loop time: ", t2 - t1)
        sorted_regions = sorted(value_regions, key=lambda x: x[1], reverse=True)
        
        return sorted_regions

    def _sorted_waypoints(self, sorted_regions: List, top_k: int=3):
        waypoints, values = [], []
        for item in sorted_regions[:top_k]:
            waypoints.append([item[2]])
            values.append(item[1])
        waypoints = np.concatenate(waypoints, axis=0)
        
        return waypoints, values
    
    def forward(self, full_map: np.ndarray, traversible: np.ndarray, value_map: np.ndarray, collision_map: np.ndarray,
                detected_classes: OrderedSet, position: Sequence, fmm_dist: np.ndarray, replan: bool, step: int, current_episode_id: int):
        if np.sum(value_map.astype(bool)) < 24**2:
            best_waypoint = np.array([int(position[0]), int(position[1])])
            best_value = 0.
            sorted_waypoints = [np.array([int(position[0]), int(position[1])])]
            
            return best_waypoint, best_value, sorted_waypoints
        else:
            # t1 = time.time()
            sorted_regions = self._get_sorted_region_fast_slic(full_map, traversible, value_map, collision_map, detected_classes)
            # t2 = time.time()
            # print("super pixel time: ", t2 - t1)
            sorted_waypoints, sorted_values = self._sorted_waypoints(sorted_regions)
            # print("sorted_waypoints and sorted values: ", sorted_waypoints, sorted_values)
            best_waypoint, best_value, sorted_waypoints = \
                self.waypoint_selector(sorted_waypoints, position, collision_map, value_map, fmm_dist, traversible, replan)
                
            if self.visualize:
                self._visualize(sorted_regions, value_map, step, current_episode_id)
            # print("best value: ", best_value)
            return best_waypoint, best_value, sorted_waypoints
    
    def _visualize(self, sorted_regions: List, value_map: np.ndarray, step: int, current_episode_id: int):
        waypoints = []
        res = np.zeros(value_map.shape)
        num_regions = len(sorted_regions)
        
        for i, (mask, _, _) in enumerate(sorted_regions):
            res[mask == 1] = num_regions + 1 - i
            _, _, _, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            waypoint = np.array([int(centroids[1][1]), int(centroids[1][0])])
            waypoints.append(waypoint)
        
        min_val = np.min(res)
        max_val = np.max(res)
        normalized_values = (res - min_val) / (max_val - min_val + 1)
        normalized_values[res == 0] = 1
        res = cv2.applyColorMap((normalized_values* 255).astype(np.uint8), cv2.COLORMAP_HOT)
        for waypoint in waypoints:
            cv2.circle(res, (waypoint[1], waypoint[0]), radius=2, color=(0,0,0), thickness=-1)
        
        cv2.imshow("super pixel", np.flipud(res))
        cv2.waitKey(1)
        # if self.visualize:
            # cv2.imwrite('img_debug/super_pixel.png', np.flipud(res))
        
        if self.print_images:
            save_dir = os.path.join(self.config.RESULTS_DIR, "super_pixel/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, np.flipud(res))