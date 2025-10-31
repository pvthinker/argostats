import numpy as np
from math import prod
from pathlib import Path
from argostats.tools.json import jsonencoder, tuplify
from argostats.binaryfiles import BinaryFile, read_header, read_data


class ArrayOfStruct:
    """Array of Struct: n-dimensional mutable array of heterogeneous data.

    Each element of the n-dimensional array is a struct. The elements
    of the struct can be single values of any type (provided it is a
    known numpy.dtype) or arrays of values of any type.

    This class fills a hole not covered by either pandas.DataFrame or
    Numpy structured arrays.

    Under the hood, the data are stored in a single numpy array of "i1".

    Data can be accessed row-wise as array of fields or column-wise as
    sub-array of struct.

    """

    def __init__(self, struct, shape, atts={}, lazy=False):
        self.dtype = "AOS"
        self.struct = struct
        self.shape = shape
        self.atts = atts
        self.fields = struct.fields
        self.nbytes = prod(shape)*self.struct.size
        self.lazy = lazy
        self._add_atts()
        if not lazy:
            self.allocate()
            self._add_fields()

    def _add_atts(self):
        for k, v in self.atts.items():
            setattr(self, k, v)

    def _add_fields(self):
        for field in self.fields:
            setattr(self, field, self.__getitem__(field))

    def zeros_like(self, shape):
        return ArrayOfStruct(self.struct, shape, atts=self.atts)

    @property
    def size(self):
        return prod(self.shape)

    @property
    def infos(self):
        properties = {"dtype": self.dtype,
                      "shape": self.shape,
                      "struct size": self.struct.size}

        return {"properties": properties,
                "struct": self.struct.variables,
                "atts": self.atts}

    def tojson(self):
        return jsonencoder({"Content": self.infos})

    def __repr__(self):
        return "<Array Of Struct>\n"+jsonencoder(self.infos)

    def crop(self, idx):
        if idx.dtype == "bool":
            n = int(sum(idx))
        else:
            n = idx.size
        res = ArrayOfStruct(self.struct, (n,), atts=self.atts, lazy=self.lazy)
        if not self.lazy:
            res.data[:, :] = self.data[idx, :]
        return res

    def add_columns(self, variables):
        if count(variables) > 1:
            newvariables = self.struct.variables+tuplify(variables)
        else:
            newvariables = self.struct.variables+(variables,)

        newstruct = Struct(newvariables)
        res = ArrayOfStruct(newstruct, self.shape,
                            atts=self.atts, lazy=self.lazy)

        if not self.lazy:
            n = self.data.shape[-1]
            res.data[..., :n] = self.data

        return res

    def allocate(self):
        if not self.lazy:
            self.data = np.zeros(self.shape+(self.struct.size,), dtype="i1")

    def __getitem__(self, name):
        if isinstance(name, list) or isinstance(name, tuple):
            return [self.__getitem__(n) for n in name]

        assert not self.lazy, "Not possible for lazy AOS"
        content = self.struct.content
        assert name in content, f"{name} not in {self.fields}"
        dtype, offset, nbytes, shape = content[name]
        res = self.data[..., slice(offset, offset+nbytes)].view(dtype)
        res.shape = self.shape+shape
        return res

    def to_binary(self, filename):
        """Write the AOS in a Binary File

        The AOS is recovered with
        >>> aos = read_binary(filename)
        """
        binfile = BinaryFile(filename, self.tojson(), self.nbytes)
        if not self.lazy:
            binfile.write_data(self.data)
        return binfile


def count(variables):
    assert isinstance(variables, tuple)
    if isinstance(variables[0], tuple):
        return len(variables)
    else:
        return 1


class Struct:
    """Define the structure for the Array of Struct

    Parameters
    ----------
    variables: tuple of tuple
        each inner tuple is of the form

        ("NAME", dtype) or ("NAME, dtype, n) or ("NAME, dtype, shape)

        where
        - "NAME", str, name of the field
        - dtype , str, is a numpy dtype, see np.typecodes
        - n, int, the length of the field
        - shape, tuple of int, the shape of the field

        in the case of ("NAME", dtype) the field has a one-element
    """

    def __init__(self, variables):
        self.variables = tuplify(variables)
        self.set_structure()
        self.fields = list(self.content.keys())

    def __repr__(self):
        infos = {
            "Struct": {
                "size": self.size,
                "fields": self.fields}
        }
        return jsonencoder(infos)

    def set_structure(self):
        offset = 0
        content = {}
        for var in self.variables:
            name, dtype, nelem, shape = get_nelem(var)
            nbytes = np.dtype(dtype).itemsize*nelem
            content[name] = (dtype, offset, nbytes, shape)
            offset += nbytes
        self.size = offset
        self.content = content


def get_nelem(var):
    if len(var) == 2:
        name, dtype = var
        return (name, dtype, 1, ())

    elif len(var) == 3:
        name, dtype, shape = var
        if isinstance(shape, int):
            nelem = shape
            return (name, dtype, nelem, (nelem,))

        else:
            return name, dtype, prod(shape), shape

    raise ValueError


def read_binary(filename):
    """ Read a binary file (*.bin) file into an ArrayOfStruct.
    """

    fname = Path(filename)
    assert fname.is_file()
    header, data_offset, ndatabytes = read_header(fname)
    data = read_data(fname, 0, ndatabytes)

    specs = header["Content"]
    variables = specs["struct"]
    atts = specs["atts"]
    shape = specs["properties"]["shape"]

    struct = Struct(variables)
    aos = ArrayOfStruct(struct, shape, atts=atts)
    aos.data[...] = data.reshape(aos.data.shape)
    return aos


def write_aos_chunk_to_binaryfile(binfile, aos_chunk, startindex):
    location = aos_chunk.struct.size*startindex
    data = aos_chunk.data
    print(f"write chunk size={data.size} @ index={startindex}")
    binfile.write_data_chunk(data, location)
