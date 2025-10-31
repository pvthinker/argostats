import numpy as np
import json
from pathlib import Path
from argostats.tools.json import jsonencoder
from argostats.aos import ArrayOfStruct, Struct, read_binary, write_aos_chunk_to_binaryfile
from argostats.binaryfiles import inttobyte, bytetoint
import datetime
import os


def test_simplebinaryfile():

    filename = Path("test.bin")
    d = {"variables": ["T", "S"],
         "pressure levels": list(np.linspace(0, 10, 6)),
         "number of profiles": 1234}

    metadata = jsonencoder(d)
    header = f"\nInterpolated Argo Profiles\n{metadata}\n"

    data = np.arange(10, dtype="f4").view("i1")

    offset = 4+len(header)

    # write
    with open(filename.as_posix(), "bw") as fid:
        fid.seek(0)
        fid.write(inttobyte(offset))
        fid.write(header.encode('utf-8'))
        fid.write(data.tobytes())

    # read
    with open(filename.as_posix(), "br") as fid:
        offset = bytetoint(fid.read(4))
        header = fid.read(offset)
        # print(f"offset={offset}, header={header}")

    filesize = os.stat(filename).st_size
    datasize = filesize-offset
    res = np.fromfile(filename,
                      count=datasize,
                      offset=offset,
                      dtype="i1")

    # decode
    dico = json.loads(
        "\n".join(header.decode("utf-8").split("\n")[2:-1]))

    # check read/write is reversible
    assert np.allclose(res, data)
    assert dico == d

    return dico, res


def get_aos_example(N, nlevs):

    DACS = ["aoml", "bodc", "coriolis", "csio", "csiro",
            "incois", "jma", "kiost", "kma", "meds", "nmdis"]

    pref = np.linspace(0, nlevs*10, nlevs)
    atts = {"PREF": list(pref),
            "DACS": DACS}

    variables = [["GINDEX", "i4"],
                 ["WMO", "i4"],
                 ["DAC", "i1"],
                 ["IPROF", "i2"],
                 ["JULD", "<M8[s]"],
                 ["LATITUDE", "f4"],
                 ["LONGITUDE", "f4"],
                 ["FLAG", "i1"],
                 ["NVALUES", "i2"],
                 ["TEMP", "f4", nlevs],
                 ["PSAL", "f4", nlevs],
                 ["IDX", "i1", nlevs]
                 ]

    s = Struct(variables)

    aos = ArrayOfStruct(s, (N,), atts=atts)
    aos["GINDEX"][:] = np.arange(N)
    T = aos["TEMP"]
    T[0, :] = 9
    T[1, :] = np.arange(nlevs)
    T[-1, :] = 99
    aos["LATITUDE"][:] = np.linspace(-80, 80, N)
    JULD = aos["JULD"]
    JULD[:] = datetime.datetime(2025, 10, 25)

    return aos


def test_aos():
    N = 10
    nlevs = 4
    q = get_aos_example(N, nlevs)
    assert q.shape == (N,)
    assert q["TEMP"][1, -1] == nlevs-1
    assert q["LATITUDE"][0] == -80

    r = q.add_columns(("RHO", "f4", nlevs))
    t = q.crop(q["LATITUDE"] > 0)

    assert all(t["LATITUDE"] > 0)
    assert t.nbytes < q.nbytes
    assert r.nbytes == q.nbytes+N*nlevs*4
    assert t.fields == q.fields
    assert "RHO" in r.fields
    assert "RHO" not in q.fields


def test_binaryfile():
    N = 1000
    nlevs = 20
    q = get_aos_example(N, nlevs)
    f = "toto.bin"
    binf = q.to_binary(f)

    aos = read_binary(f)

    assert np.allclose(q.data, aos.data)
    assert aos.fields == q.fields


def test_lazyAOS_and_chunk():
    N = 10
    nlevs = 4
    q = get_aos_example(N, nlevs)
    f = "toto.bin"

    aosbig = ArrayOfStruct(q.struct, (1000,), atts=q.atts, lazy=True)

    binfile = aosbig.to_binary(f)
    small = aosbig.zeros_like((N,))
    small.data[:] = q.data

    startindex = 957

    write_aos_chunk_to_binaryfile(binfile, small, startindex)

    z = read_binary(f)
    assert np.allclose(z.data[startindex:startindex+N, :], q.data)
