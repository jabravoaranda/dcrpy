from pathlib import Path

import yaml
from .types import RadarInfoType


def _read_yaml(path: Path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


RADAR_INFO: RadarInfoType = _read_yaml(Path(__file__).parent.absolute() / "info.yml")
RADAR_PLOT_INFO = _read_yaml(Path(__file__).parent.absolute() / "plotting" / "info.yml")

__version__ = "0.1.0"

__all__ = ["RADAR_PLOT_INFO", "RADAR_INFO", "rpg_nc", "utils", "plotting", "types", "nebula"]


__doc__ = """The top of the radar module that is compatible (at least) with GFAT radar: "NEPHELE" AND "NEBULA"

    Contributors:
    
    -   Juan Antonio Bravo Aranda (Ph.D.)
    -   Leonie von Terzi (Ph.D.)
    -   Matheus Tolentino Da Silva
    
"""
