"""
Default environment, simulaotr, agent configs
"""
from typing import List, Optional, Union

from habitat.config.default import Config as CN
from habitat.config.default import get_config


_C = get_config()
_C.defrost()

# -----------------------------------------------------------------------------
# VLN SENSOR POSE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SENSOR_POSE_SENSOR = CN()
_C.TASK.SENSOR_POSE_SENSOR.TYPE = "SensorPoseSensor"

# -----------------------------------------------------------------------------
# VLN LLM SENSOR
# -----------------------------------------------------------------------------
_C.TASK.LLM_SENSOR = CN()
_C.TASK.LLM_SENSOR.TYPE = "LLMSensor"

# -----------------------------------------------------------------------------
# VLN POSITION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POSITION_SENSOR = CN()
_C.TASK.POSITION_SENSOR.TYPE = "PositionSensor"

# -----------------------------------------------------------------------------
# VLN HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"

# ----------------------------------------------------------------------------
# NDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.NDTW = CN()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_unseen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 3.0

# ----------------------------------------------------------------------------
# SDTW MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.SDTW = CN()
_C.TASK.SDTW.TYPE = "SDTW"

# ----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"

# -----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = CN()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"

# -----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0

# -----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = CN()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
_C.TASK.ORACLE_SPL.SUCCESS_DISTANCE = 3.0

# -----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"

# ----------------------------------------------------------------------------
# POSITION MEASUREMENT For faster eval
# ----------------------------------------------------------------------------
_C.TASK.POSITION = CN()
_C.TASK.POSITION.TYPE = 'Position'

_C.DATASET.split_num = 0
_C.DATASET.split_rank = 0

# -----------------------------------------------------------------------------
# TOP_DOWN_MAP_VLNCE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP_VLNCE = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.TYPE = "TopDownMapVLNCE"
_C.TASK.TOP_DOWN_MAP_VLNCE.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP_VLNCE.MAP_RESOLUTION = 512
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_BORDER = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_SHORTEST_PATH = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_FIXED_WAYPOINTS = False
_C.TASK.TOP_DOWN_MAP_VLNCE.DRAW_MP3D_AGENT_PATH = False
_C.TASK.TOP_DOWN_MAP_VLNCE.GRAPHS_FILE = "data/connectivity_graphs.pkl"
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.DRAW = False
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.FOV = 79
_C.TASK.TOP_DOWN_MAP_VLNCE.FOG_OF_WAR.VISIBILITY_DIST = 5.0

# ----------------------------------------------------------------------------
# DATASET EXTENSIONS
# ----------------------------------------------------------------------------
_C.DATASET.EPISODES_ALLOWED = [701]
# _C.DATASET.EPISODES_ALLOWED = [389]


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
    config.freeze()
    return config