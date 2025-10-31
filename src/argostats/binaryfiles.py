import numpy as np
import os
from pathlib import Path
import datetime
import subprocess
import json
from argostats.tools.json import jsonencoder, tuplify


class BinaryFile:
    """File format to store data in binary mode with a self description.

    This format is for speed, size and efficiency.

    The description of the data is stored in plain ascii text in the
    header of the file (json format). The data are stored after the
    header. The plain ascii header allows to visually check the
    content of the file with shell commands like more or head.

    This format imposes to know the size (in bytes) of the data at the
    creation of the file. The file is created immediately with the
    proper size, somehow allocating on disk the space.

    Data are written afterwards, in one call, or by chunks.

    """

    def __init__(self, filename, header, ndatabytes, overwrite=True):
        self.filename = Path(filename)
        self._header = header
        self.ndatabytes = ndatabytes
        self.create(overwrite)
        self.write_header()

    def __repr__(self):
        s = ["Binary File"]
        s += [f" - name: {self.filename}"]
        s += [f" - size: {self.filesize}"]
        s += [f" - header size: {len(self.header)}"]
        return "\n".join(s)

    @property
    def header(self):
        return self._header

    @property
    def data_offset(self):
        return 4+len(self.header)

    @property
    def filesize(self):
        return self.data_offset+self.ndatabytes

    def create(self, overwrite):
        f = self.filename
        if f.is_file():
            if overwrite:
                print(f"Warning {f} is overwritten")
                os.remove(f)
            else:
                print(f"Warning {f} already exists")
                print("rename it before creating a new one")
                return
        fsize = self.filesize
        print(f"create {f}")
        print(f"   - size  : {fsize} Bytes")
        create_empty_file(f, fsize)

    def write_data(self, data):
        assert self.data_offset+data.size*data.dtype.itemsize == self.filesize

        with open(self.filename.as_posix(), "br+") as fid:
            fid.seek(self.data_offset)
            n = fid.write(data.tobytes())
            assert n == data.size

    def write_data_chunk(self, data, location):
        assert location >= 0
        msg = f"Try to write data chunk beyond the end of file: {self.filename}"
        end = self.data_offset+location+data.nbytes
        assert end <= self.filesize, msg

        with open(self.filename.as_posix(), "br+") as fid:
            fid.seek(self.data_offset+location)
            fid.write(data.tobytes())

    def write_header(self):
        with open(self.filename.as_posix(), "br+") as fid:
            fid.seek(0)
            fid.write(inttobyte(self.data_offset))
            fid.write(self.header.encode('utf-8'))


def read_header(filename):
    with open(filename.as_posix(), "br") as fid:
        data_offset = bytetoint(fid.read(4))
        header = fid.read(data_offset-4).decode("utf-8")

    filesize = os.stat(filename).st_size
    ndatabytes = filesize-data_offset
    try:
        metadata = json.loads(header)
    except:
        raise ValueError("Header file is corrupted: not in JSON format")

    return tuplify(metadata), data_offset, ndatabytes


def read_data(filename, location, ndatabytes):
    with open(filename.as_posix(), "br") as fid:
        data_offset = bytetoint(fid.read(4))

    return np.fromfile(filename,
                       count=ndatabytes,
                       offset=data_offset+location,
                       dtype="i1")


def create_empty_file(filename, filesize):
    command = f"dd if=/dev/zero of={filename} bs=1 count=0 seek={filesize}"
    a = subprocess.run(command.split(),
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def bytetoint(b): return int.from_bytes(b, byteorder="little", signed=True)
def inttobyte(i): return int.to_bytes(i, 4, byteorder="little", signed=True)

# def convert_oldbin_to_binary(fnew):
#     from argostats.toctools import load_argo_summary
#     from argostats.interpolation import get_64pref
#     df = load_argo_summary()
#     APF = ArgoProfilesFile(df)
#     P = APF.coding_scheme
#     data = APF.read_all()

#     variables = (('WMO', 'i4'),
#                  ('IPROF', 'i4'),
#                  ('JULD', '<M8[s]'),
#                  ('LATITUDE', 'f4'),
#                  ('LONGITUDE', 'f4'),
#                  ('NVALUES', 'i4'),
#                  ('UNUSED', 'i4'),
#                  ('TEMP', 'f4', 64),
#                  ('PSAL', 'f4', 64),
#                  ('IDX', 'i1', 64))

#     DACS = ["aoml", "bodc", "coriolis", "csio", "csiro",
#             "incois", "jma", "kiost", "kma", "meds", "nmdis"]
#     atts = {"PREF": list(get_64pref()),
#             "DACS": DACS}

#     s = Struct(variables)

#     N = len(df)
#     aos = ArrayOfStruct(s, (N,), atts=atts)
#     aos.data[:, :] = data.reshape(aos.data.shape)
#     aos.JULD[:] = df.JULD.astype("datetime64[s]")

#     aos.to_binary(fnew)
