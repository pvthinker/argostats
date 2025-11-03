import xarray as xr
import numpy as np
import os
import pandas as pd
import datetime
from functools import reduce
import subprocess
from pathlib import Path

from argostats.tools.parallel import get_pool


GDAC = os.environ["GDAC"]
ARGOSTATS = os.environ["ARGOSTATS"]

assert Path(GDAC).is_dir()
assert Path(ARGOSTATS).is_dir()

TOCFILE = f"{ARGOSTATS}/argo_toc.pickle"


DACS = ["aoml", "bodc", "coriolis", "csio", "csiro",
        "incois", "jma", "kiost", "kma", "meds", "nmdis"]

DATA_MODES = [b"R", b"D", b"A"]

PLATFORMS = [
    b'                                ',
    b'ALTO                            ',
    b'APEX                            ',
    b'APEX-SBE                        ',
    b'APEX_D                          ',
    b'ARVOR                           ',
    b'ARVOR_C                         ',
    b'ARVOR_D                         ',
    b'HM2000                          ',
    b'HM4000                          ',
    b'NAVIS                           ',
    b'NAVIS_A                         ',
    b'NAVIS_EBR                       ',
    b'NEMO                            ',
    b'NINJA                           ',
    b'NINJA_D                         ',
    b'NOVA                            ',
    b'NOVA-SBE                        ',
    b'Nova-SBE                        ',
    b'PALACE                          ',
    b'POPS_PROVOR                     ',
    b'PROVOR                          ',
    b'PROVOR-SBE                      ',
    b'PROVOR_II                       ',
    b'PROVOR_III                      ',
    b'PROVOR_IV                       ',
    b'PROVOR_MT                       ',
    b'PROVOR_V                        ',
    b'PROVOR_V_JUMBO                  ',
    b'Primary sampling: averaged []   ',
    b'S2A                             ',
    b'S2X                             ',
    b'SOLO                            ',
    b'SOLO-W                          ',
    b'SOLO_BGC                        ',
    b'SOLO_BGC_MRV                    ',
    b'SOLO_D                          ',
    b'SOLO_D_MRV                      ',
    b'SOLO_II                         ',
    b'SOLO_W                          ',
    b'XUANWU                          ',
    b'n/a                             ']


def load_summary():
    f = Path(TOCFILE)
    if f.is_file():
        return pd.read_pickle(TOCFILE)
    else:
        print("You need to first build the Argo Summary")
        print(">>> build_argo_summary()")
        print("This will take about 5 minutes")


def save_summary(df):
    df.to_pickle(TOCFILE)


def build_summary():
    dacwmos = get_allwmos()
    df = get_infos(dacwmos)
    print(f"save Argo Summary in {TOCFILE}")
    save_summary(df)


def get_dacwmos_summary(dacwmos):
    return {dac: len(wmos) for dac, wmos in dacwmos.items()}


def get_allwmos():
    with get_pool() as pool:
        dacwmos = pool.map(get_dacwmos, DACS)

    return {dac: wmos for dac, wmos in dacwmos}


def get_dacwmos(dac):
    dirs = Path(dirdac(dac)).glob("*")
    return (dac, [int(d.name) for d in dirs])


def dirdac(dac):
    return f"{GDAC}/{dac}"


def dirwmo(dac, wmo):
    return f"{GDAC}/{dac}/{wmo}"


def file_prof(dac, wmo):
    return f"{GDAC}/{dac}/{wmo}/{wmo}_prof.nc"


def get_infos_from_single_dac(dacwmos, dac="kiost"):
    return get_infos({dac: dacwmos[dac]})


def count_wmos(dacwmos):
    return sum((n
                for d, n in get_dacwmos_summary(dacwmos).items()))


def get_infos(dacwmos, test=False):
    wmos_list = ((dac, wmo)
                 for dac in dacwmos
                 for wmo in dacwmos[dac])

    if not test:
        print("collect infos of wmos")
        print(f"#wmos: {count_wmos(dacwmos)}")
        print("---------------------")
        print(" dac           wmo")
        print("---------------------")

    with get_pool() as pool:
        infos = pool.starmap(get_wmo_infos, wmos_list)

    return pd.concat(infos)


def get_infos_dtype(infos):
    return [(c, infos[c].dtype)
            for c in infos]


def get_wmo_infos(dac, wmo):
    print(f"\r{dac:10}   {wmo}", end="")
    ds = get_dataset(dac, wmo)

    if ds is None:
        return None

    names = ["LONGITUDE", "LATITUDE"]

    infos = {name: np.float32(getattr(ds, name).data)
             for name in names}

    infos["JULD"] = getattr(ds, "JULD")

    sizes = ["N_LEVELS", "N_PROF"]

    for size in sizes:
        infos[size] = np.int16(len(getattr(ds, size)))

    n = len(ds.LONGITUDE)
    infos["DAC"] = np.int8(DACS.index(dac))
    infos["WMO"] = wmo
    infos["IPROF"] = np.arange(n, dtype="i2")
    infos["DATA_MODE"] = _data_mode(ds.DATA_MODE)
    infos["PLATFORM_TYPE"] = _platform(ds.PLATFORM_TYPE.data)
    names = ["POSITION", "JULD"]
    qc_list = (ds[f"{n}_QC"].data.astype("e") for n in names)

    infos["FLAG"] = combine_qc(qc_list)

    return pd.DataFrame(infos)


def combine_qc(qc_list):
    return reduce(lambda x, y: x*y, qc_list).astype("i1")


def get_dataset(dac, wmo):
    ncfile = Path(file_prof(dac, wmo))
    return xr.open_dataset(ncfile) if ncfile.is_file() else None


_platform = np.vectorize(lambda x: np.int8(PLATFORMS.index(x)))
_data_mode = np.vectorize(lambda x: np.int8(DATA_MODES.index(x)))


def get_nprof(df):
    return df.groupby("WMO", group_keys=True).first().N_PROF
