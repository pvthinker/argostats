import numpy as np
import gsw
from scipy import interpolate
from functools import reduce
from pathlib import Path

from argostats.binaryfiles import BinaryFile, read_data
from argostats.aos import ArrayOfStruct, Struct, read_binary
from argostats.tools.parallel import get_pool, chrono
from argostats.toctools import DACS, ARGOSTATS, get_dataset


def basic_interpolation(SA, CT, PRES, PRES_i):
    kwargs = {"kind": "cubic", "fill_value": "extrapolate"}
    sa = interpolate.interp1d(PRES, SA, **kwargs)
    ct = interpolate.interp1d(PRES, CT, **kwargs)
    return sa(PRES_i), ct(PRES_i)


METHODS = {0: basic_interpolation,
           1: gsw.interpolation.sa_ct_interp}

MINIMAL_NUMBER_DATA_IN_PROFILE = 10

ARGOSTRUCT = (('WMO', 'i4'),
              ('DAC', 'i1'),
              ('IPROF', 'i2'),
              ('JULD', '<M8[s]'),
              ('LATITUDE', 'f4'),
              ('LONGITUDE', 'f4'),
              ('DATA_MODE', 'i1'),
              ('FLAG', 'i1'),
              ('NVALUES', 'i4'),
              ('CT', 'f4', 64),
              ('SR', 'f4', 64),
              ('IDX', 'i1', 64))

FILE_PROFILES = "allprofiles.bin"


def load_profiles():
    f = Path(f"{ARGOSTATS}/{FILE_PROFILES}")
    if f.is_file():
        print(f"Load interpolated profiles {f}")
        return read_binary(f)
    else:
        print(f"Interpolated profiles not found")
        print(f"{f} does not exist")


class ArgoInterp:
    def __init__(self, df, algo=0):
        self.df = df
        self.df.index = np.arange(len(df))
        self.PREF = pref64()
        self.AOS = ArrayOfStruct(Struct(ARGOSTRUCT),
                                 (len(df),),
                                 atts={"PREF": list(self.PREF),
                                       "DACS": DACS},
                                 lazy=True)
        self.binf = self.create_binaryfile()
        self.algo = algo

    def create_binaryfile(self):
        filename = Path(f"{ARGOSTATS}/{FILE_PROFILES}")
        return BinaryFile(filename,
                          self.AOS.tojson(),
                          self.AOS.nbytes,
                          overwrite=False)

    def location_in_file(self, profindex):
        return profindex*self.AOS.struct.size

    @ property
    def method(self):
        if self.algo == 0:
            return basic_interpolation
        elif self.algo == 1:
            return gsw.interpolation.sa_ct_interp
        else:
            raise ValueError

    def write_header(self):
        aos = self.AOS.zeros_like((len(self.df),))
        print("load binary file")
        aos.data[:, :] = read_data(self.binf.filename,
                                   self.location_in_file(0),
                                   aos.nbytes
                                   ).reshape(aos.data.shape)

        variables = ['WMO', 'DAC', 'IPROF', 'JULD',
                     'LATITUDE', 'LONGITUDE', 'DATA_MODE', 'FLAG']

        print("set header variables")
        content = self.AOS.struct.content
        for v in variables:
            dtype = content[v][0]
            aos[v][:] = self.df[v].astype(dtype)

        print("update binary file")
        self.binf.write_data_chunk(aos.data,
                                   self.location_in_file(0)
                                   )

    def proceed_single_wmo(self, wmo):
        dfwmo = self.df[self.df.WMO == wmo]

        good_profiles = [k for k, f in enumerate(dfwmo.FLAG) if f == 1]
        if len(good_profiles) == 0:
            return

        dac = retrieve_dac(self.df, wmo)
        raw_profiles = load_wmo_profiles_from_netcdf(dac, wmo)
        if raw_profiles is None:
            return

        add_eos10_variables(raw_profiles)

        aos = self.AOS.zeros_like((len(dfwmo),))
        variables = ["NVALUES", "CT", "SR", "IDX"]
        NVAL, CT, SR, IDX = aos[variables]

        for iprof in good_profiles:

            raw_profile = extract_raw_profile(raw_profiles, iprof)

            idx, sr, ct = interpolate_profile(
                raw_profile, self.PREF, self.method)

            NVAL[iprof] = len(idx)
            if NVAL[iprof] > 0:
                CT[iprof, idx] = ct.astype(CT.dtype)
                SR[iprof, idx] = sr.astype(SR.dtype)
                IDX[iprof, idx] = 1

        self.binf.write_data_chunk(aos.data,
                                   self.location_in_file(dfwmo.index[0])
                                   )

        print(f"\rwmo {wmo:6} | valid {len(good_profiles):4}")

    @ chrono
    def proceed_all(self, wmos=None):
        if wmos is None:
            wmos = self.df.WMO.unique()

        with get_pool() as pool:
            wmo_index = get_wmo_sweep(len(wmos), pool._processes)
            pool.map(self.proceed_single_wmo, (wmos[k] for k in wmo_index))

        self.write_header()

    def read_wmo(self, wmo):
        dfwmo = self.df[self.df.WMO == wmo]

        aos = self.AOS.zeros_like((len(dfwmo),))

        aos.data[:, :] = read_data(self.binf.filename,
                                   self.location_in_file(dfwmo.index[0]),
                                   aos.nbytes
                                   ).reshape(aos.data.shape)
        return aos


def get_wmo_sweep(nwmos, nprocs):
    n = nwmos//nprocs+1
    i = np.arange(nprocs*n).reshape(nprocs, n)
    idx = i.T.ravel()
    return idx[idx < nwmos]


def extract_raw_profile(data, iprof):
    d = {}
    idx, = np.where(data["QC"][iprof] == 1)
    if len(idx) < 5:
        return None
    for n in ["PRES", "TEMP", "PSAL", "QC"]:
        d[n] = data[n][iprof, idx]
    return d


def load_wmo_profiles_from_netcdf(dac, wmo):
    ds = get_dataset(DACS[dac], wmo)
    data = {}
    names = ["TEMP", "PRES", "PSAL"]
    if not all([n in ds for n in names]):
        return None
    DATA_MODE = ds.DATA_MODE.values.astype('S1')
    if all([x == b'D' for x in DATA_MODE]):
        names_in = ["TEMP_ADJUSTED", "PRES_ADJUSTED", "PSAL_ADJUSTED"]
    else:
        names_in = names
    for name, name_in in zip(names, names_in):
        data[name] = ds[name_in].data
    data["QC"] = reduce(
        lambda x, y: x*y, [ds[f"{n}_QC"].data.astype("e") for n in names_in])
    return data


def add_eos10_variables(raw_profiles):
    """ Add SR and CT by mutating the input

    input: dict with PSAL, TEMP and PRES keywords
    output: same dict with new keywords CT and SR

    """
    SR = gsw.SR_from_SP(raw_profiles["PSAL"])
    CT = gsw.CT_from_t(SR, raw_profiles["TEMP"], raw_profiles["PRES"])
    raw_profiles["SR"] = SR
    raw_profiles["CT"] = CT


def interpolate_profile(ds, PREF, method):
    if ds is None:
        return [], None, None

    names = ["TEMP", "PRES", "PSAL"]
    if not all([n in ds for n in names]):
        return [], None, None

    SA, CT, PRES, qc = ds["PSAL"], ds["TEMP"], ds["PRES"], ds["QC"]
    PRES[qc != 1] = -999
    jdx, idx = get_valid_profile_data(PRES, PREF)

    if len(idx) >= MINIMAL_NUMBER_DATA_IN_PROFILE:
        sa_i, ct_i = method(SA[jdx], CT[jdx], PRES[jdx], PREF[idx])
        return idx, sa_i, ct_i
    else:
        return [], None, None

    return


def get_valid_profile_data(PRES, PRES_i):
    n, ni = len(PRES), len(PRES_i)
    if n < 5:
        return [], []

    _, jdx = np.unique(PRES, return_index=True)

    if (len(jdx) == (jdx[-1]-jdx[0]+1)) and (len(jdx) > 10):
        deltamin = 1.2*(PRES[1]-PRES[0])
        deltamax = 0  # 1.1*(PRES[n-1]-PRES[n-2])
        idx, = np.where((PRES.min()-deltamin <= PRES_i) &
                        (PRES_i <= PRES.max()+deltamax))

        return jdx, idx
    else:
        return [], []


def retrieve_dac(df, wmo):
    return int(df[df.WMO == wmo].iloc[0].DAC)


def pref64():
    return np.array([0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
                     100., 110., 120., 130., 140., 150., 160., 170.,
                     180., 190., 200., 220., 240., 260., 280, 300.,
                    320., 340., 360., 380., 400., 450., 500., 550.,
                    600., 650., 700., 750., 800., 850., 900., 950.,
                    1000., 1050., 1100., 1150., 1200., 1250., 1300.,
                    1350., 1400., 1450., 1500., 1550., 1600., 1650.,
                    1700., 1750., 1800., 1850., 1900., 1950.,
                    2000.])
