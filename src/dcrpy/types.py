from enum import Enum
from typing import TypedDict

from typing_extensions import NotRequired


class RadarName(str, Enum):
    nph = "NEPHELE"
    nbl = "NEBULA"    

class MeasurementType(str, Enum):
    zen = "ZEN"
    ppi = "PPI"
    rhi = "RHI"

class _RadarInfo(TypedDict):
    NEPHELE: dict
    NEBULA: dict

class _RadarMetadata(TypedDict):
    nick2name: dict
    name2nick: dict
    measurement_type: list

class RadarInfoType(TypedDict):
    radars: _RadarInfo
    metadata: _RadarMetadata


class ParamsDict(TypedDict):
    k_radar: float
    # particle_alpha: np.ndarray
    # particle_alpha_raman: NotRequired[np.ndarray]
    # particle_beta: np.ndarray
    # particle_beta_raman: NotRequired[np.ndarray]
    # molecular_alpha: np.ndarray
    # molecular_alpha_raman: NotRequired[np.ndarray]
    # molecular_beta: np.ndarray
    # molecular_beta_raman: NotRequired[np.ndarray]
    # particle_accum_ext: np.ndarray
    # particle_accum_ext_raman: NotRequired[np.ndarray]
    # molecular_accum_ext: np.ndarray
    # molecular_accum_ext_raman: NotRequired[np.ndarray]
    # molecular_beta_att: np.ndarray
    # transmittance: np.ndarray
    # overlap: np.ndarray
    # angstrom_exponent_fine: float
    # angstrom_exponent_coarse: float
