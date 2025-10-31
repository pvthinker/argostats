from argostats.toctools import GDAC, ARGOSTATS
import argostats.toctools as toc
from pathlib import Path
import numpy as np


def test_gdac_is_valid():
    assert Path(GDAC).is_dir()


def test_dirstats():
    assert Path(ARGOSTATS).is_dir()


DACWMOS = {"kiost": [2903935,  2900449,  2900201,
                     5900691,  2900204,  7900119,  2901779]}

dac = "kiost"


def test_get_dacwmos():
    res = toc.get_dacwmos(dac)

    wmos = res[1]
    dacwmos = {dac: wmos}

    assert res[0] == dac
    assert len(wmos) > 0

    wmo = wmos[0]
    f = Path(toc.file_prof(dac, wmo))
    assert f.is_file()


def test_get_infos():
    dacwmos = DACWMOS
    wmos = dacwmos[dac]

    assert toc.count_wmos(dacwmos) > 0

    infos = toc.get_infos(dacwmos, test=True)
    assert len(infos) > 0
    assert "JULD" in infos
    assert "LATITUDE" in infos
    assert "IPROF" in infos

    NPROFS = toc.get_nprof(infos)
    assert len(NPROFS) == len(wmos)


def test_dataset():
    wmo = DACWMOS[dac][-1]

    ds = toc.get_dataset(dac, wmo)

    assert hasattr(ds, "variables")
    assert hasattr(ds, "TEMP")

    assert toc.combine_qc(np.asarray([1, 1, 0, 1])) == 0
    assert toc.combine_qc(np.asarray([1, 2, 3, 4, 5])) > 0
