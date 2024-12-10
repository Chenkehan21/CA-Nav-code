"""
Implement other sensors if needed.
"""

import numpy as np
from gym import spaces
from typing import Any, Dict

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, Simulator, SensorTypes

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

from habitat_extensions.pose_utils import get_pose_change, get_start_sim_location, get_sim_location


@registry.register_sensor
class SensorPoseSensor(Sensor):
    """It is a senor to get sensor's pose
    """
    def __init__(self, sim: Simulator, config: Config,  *args: Any, **kwargs: Any) -> None:
        super().__init__(config=config)
        self._sim = sim
        self.episode_start = False
        self.last_sim_location = get_sim_location(self._sim)
        self.sensor_pose = [0., 0., 0.] # initialize last sensor pose as [0,0,0]
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "sensor_pose"
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-100., high=100, shape=(3,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        if not self.episode_start:
            start_position = episode.start_position
            start_rotation = np.quaternion(episode.start_rotation[-1], *episode.start_rotation[:-1])
            self.last_sim_location = get_start_sim_location(start_position, start_rotation)
            self.episode_start = True
        dx, dy, do, self.last_sim_location = get_pose_change(self._sim, self.last_sim_location)
        self.sensor_pose = [dx, dy, do]
        
        return self.sensor_pose
    

@registry.register_sensor
class LLMSensor(Sensor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        pass
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "llm_reply"
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TEXT
    
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Space:
        return spaces.Discrete(0)
    
    def get_observation(self, observations, *args: Any, episode, **kwargs: Any) -> Dict:
        return episode.llm_reply
    

@registry.register_sensor
class PositionSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    cls_uuid: str = "position"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        return self._sim.get_agent_state().position.astype(np.float32)
    
    
@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(quat, direction_vector)
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        heading = self._quat_to_xy_heading(rotation_world_agent.inverse())
        self._sim.record_heading = heading

        return heading