from pathlib import Path

from . import rpg_nc, utils, types
from gfatpy.utils.io import read_yaml
from .plot import RADAR_PLOT_INFO
from .types import RadarInfoType


RADAR_INFO: RadarInfoType = read_yaml(Path(__file__).parent.absolute() / 'info.yml')

__all__ = ["RADAR_PLOT_INFO", "RADAR_INFO", "rpg_nc", "utils", "plot", "types"]


__doc__ = """The top of the radar module that is compatible (at least) with GFAT radar: "NEPHELE" AND "NEBULA"

    Contributors:
    
    -   Dr. Leonie	von Terzi
"""
