import os
import h5py
import pyroaring as roaring
import pytoml as toml
import re
from scipy import sparse

from . import reversepolish as rpn

MAXROWS = 2**32

class MapFile:
    def __init__(self, filepath, mode='r'):
        self._mode = mode
        try:
            fmode = {
                'r': 'rb',
                'rw': 'r+b'
            }[self._mode]
        except KeyError:
            raise ValueError('invalid mode')
            
        if (not os.path.isfile(filepath)) and self._mode == 'rw':
            with open(filepath, 'wb') as fp:
                b = roaring.BitMap()
                fp.write(b.serialize())
            
        self._fp = open(filepath, fmode)
        buff = self._fp.read()
        self._fp.seek(0)
        self.map = roaring.BitMap.deserialize(buff)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def dump(self):
        buff = self.map.serialize()
        self._fp.write(buff)

    def close(self):
        if self._mode == 'rw':
            self.dump()
        self._fp.close()

class SparseColumn:
    def __init__(self, path, name):
        self.name = name
        self.shape = (1,0)
        self._paths = None
        self._set_paths(path)

    def _set_paths(self, path):
        self._paths = {
            '/': os.path.join(path, self.name),
            '/map': os.path.join(path, self.name, 'map'),
            '/data': os.path.join(path, self.name, 'data')
        }
        self._filepaths = {
            'map': os.path.join(self._paths['/map'], 'col.map'),
            'data': os.path.join(self._paths['/data'], 'col.data')
        }
        os.makedirs(self._paths['/'], exist_ok=True)
        os.makedirs(self._paths['/map'], exist_ok=True)
        os.makedirs(self._paths['/data'], exist_ok=True)

        if not os.path.isfile(self._filepaths['data']):
            with h5py.File(self._filepaths['data'], 'w') as h5f:
                h5f.create_dataset('shape', (2,), dtype='i')
                h5f['.']['shape'][:] = self.shape
                h5f.create_dataset('data', (0,), dtype='f', maxshape=(None,))

        if not os.path.isfile(self._filepaths['map']):
            with MapFile(self._filepaths['map'], 'rw') as bmf:
                pass

    def get_map(self):
        with MapFile(self._filepaths['map'], 'r') as bmf:
            return bmf.map

    def get_data(self):
        with h5py.File(self._filepaths['data'], 'r') as h5f:
            shape = h5f['.']['shape']
            data = h5f['.']['data']
            indices = self.get_map()
            indptr = (0,len(data))

            return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def _append_data(self, h5f, bmf, data, indices, shape):
        # create refs to hdf5 data
        shape0 = h5f['.']['shape']
        data0 = h5f['.']['data']

        # update hdf5 data
        l0 = len(data0)
        l = len(data)
        data0.resize((l0+l,))
        data0[l0:] = data

        self.shape = (1, max(shape[1],self.shape[1]))
        shape0[:] = self.shape

        # update map data
        bmf.map.update(indices)

    def put_data_blocks(self, blocksize, csr_blocks):
        with h5py.File(self._filepaths['data'], 'a') as h5f, MapFile(self._filepaths['map'], 'rw') as bmf:
            maxbi = 0
            for bi, b in csr_blocks:
                if b.shape[0] != 1 or b.shape[1] != blocksize:
                    raise ValueError('invalid block shape in block %d: %s != (1,%d)' % 
                        (bi, b.shape, blocksize))

                maxbi = max(maxbi, bi)
                self._append_data(h5f, bmf,
                    data = b.data,
                    indices = b.indices + bi*blocksize,
                    shape = (1, (maxbi+1)*blocksize))
        
class SparseDB:
    def __init__(self, path, name):
        self.name = name
        self._set_paths(path)

        self._init_rpn()

    def _init_rpn(self):
        self._fmtpat = re.compile(r'([\&\|\^\-\!])')

        tokeniser = lambda s: rpn.simple_tokeniser('bool', self._format(s))
        dispatcher = {
            '&': lambda x, y: x & y,
            '|': lambda x, y: x | y,
            '^': lambda x, y: x ^ y,
            '-': lambda x, y: x - y,
            '!': lambda x: x.flip(0, self.get_shape()[0])
        }
        unwrapper = lambda c: self._cols[self._colidx[c]].get_map() \
            if type(c) == str and c in self._meta['cols'] else c
        self._rpn = rpn.ReversePolish(tokeniser, dispatcher, unwrapper)
        
    def _format(self, statement):
        return ' '.join(self._fmtpat.sub(' \\1 ', statement).split())

    def _set_paths(self, path):
        self._paths = {
            '/': os.path.join(path, self.name),
            '/cols': os.path.join(path, self.name, 'cols')
        }
        self._filepaths = {
            'meta': os.path.join(self._paths['/'], 'meta.toml'),
        }
        os.makedirs(self._paths['/'], exist_ok=True)
        os.makedirs(self._paths['/cols'], exist_ok=True)

    def _read_meta(self):
        with open(self._filepaths['meta'], 'r') as fp:
            self._meta = toml.load(fp)
            if len(self._meta['cols']) != self._meta['shape'][1]:
                raise ValueError('inconsistent meta data')

    def _write_meta(self):
        with open(self._filepaths['meta'], 'w') as fp:
            toml.dump(self._meta, fp, sort_keys=True)

    def exists(self):
        return os.path.isfile(self._filepaths['meta'])

    def create(self, cols):
        if self.exists():
            raise ValueError('database already exists')

        if len(cols) != len(set(cols)):
            raise ValueError('repeated column names')

        self._meta = {
            'cols': cols,
            'shape': [0, len(cols)]
        }
        self._write_meta()
        
        self.attach()

    def attach(self):
        if not self.exists():
            raise ValueError('database does not exist')

        self._read_meta()

        self._colidx = {c:i for i,c in enumerate(self._meta['cols'])}
        self._cols = [SparseColumn(self._paths['/cols'], c) for c in self._meta['cols']]

    def find(self, statement):
        b = self._rpn.execute(statement)
        return list(b)

    def get_shape(self):
        return tuple(self._meta['shape'])

    def get_data(self, indices=None, cols=None):
        if cols is None:
            cols = self._meta['cols']
        if indices is None:
            return sparse.vstack(self._cols[self._colidx[c]].get_data() for c in cols).T
        else:
            return sparse.vstack(self._cols[self._colidx[c]].get_data()[0, indices] for c in cols).T

    def put_data_blocks(self, blocksize, csr_blocks):
        maxbi = 0
        for bi,blk in csr_blocks:
            maxbi = max(maxbi, bi)
            blkc = blk.tocsc()
            for i,c in enumerate(blkc.T):
                self._cols[i].put_data_blocks(blocksize, [(bi,c)])
            self._meta['shape'][0] = (maxbi+1)*blocksize
            self._write_meta()
