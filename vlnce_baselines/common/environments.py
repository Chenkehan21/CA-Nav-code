from typing import Any, Dict, Union

import habitat
from habitat import Config, Dataset
from habitat.core.embodied_task import Metrics
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions.pose_utils import get_sim_location


@baseline_registry.register_env(name="VLNCEZeroShotEnv")
class VLNCEZeroShotEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Union[Dataset,None]=None) -> None:
        super().__init__(config.TASK_CONFIG, dataset)
        self.sensor_pose_sensor = self.habitat_env.task.sensor_suite.get('sensor_pose')
    
    def reset(self) -> Observations:
        self.sensor_pose_sensor.episode_start = False
        self.last_sim_location = get_sim_location(self.habitat_env.sim)
        self.sensor_pose = [0., 0., 0.] # initialize last sensor pose as [0,0,0]
        obs = super().reset()
        
        return obs
    
    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        obs, reward, done, info = super().step(action, **kwargs)
        
        return obs, reward, done, info

    def get_reward(self, observations: Observations) -> float:
        return 0.0
        
    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()
    
    def get_done(self, observations):
        return self._env.episode_over
    
    def get_reward_range(self):
        return (0.0, 0.0)
    
    def get_reward(self, observations: Observations) -> Any:
        return 0.0
    
    def _get_sensor_pose(self):
        pass