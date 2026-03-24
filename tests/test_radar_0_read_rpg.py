from pathlib import Path
from rpgpy import read_rpg


LV0_FILE = Path(r"tests\data\RAW\nebula_w\2024\03\13\240313_150001_P00_ZEN.LV0")

def test_read_rpg():
    header, data = read_rpg(LV0_FILE)

    assert isinstance(header, dict)
    assert isinstance(data, dict)
