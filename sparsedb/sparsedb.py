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
        self._maxrows = MAXROWS
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
            with h5py.File(self._filepaths['data'], 'w') as f:
                for o,s,dt in ( ('data',(0,),'f'), ('indices',(0,),'i'), ('indptr',(2,),'i'), ('shape',(2,),'i')):
                    f.create_dataset(o, s, dtype=dt, maxshape=(None,))

        if not os.path.isfile(self._filepaths['map']):
            with MapFile(self._filepaths['map'], 'rw') as bmf:
                pass

    def get_map(self):
        with MapFile(self._filepaths['map'], 'r') as bmf:
            return bmf.map

    def get_data(self):
        with h5py.File(self._filepaths['data'], 'r') as h5f:
            data0 = h5f['.']['data']
            indices0 = h5f['.']['indices']
            indptr0 = h5f['.']['indptr']
            shape0 = h5f['.']['shape']

            return sparse.csr_matrix((data0, indices0, indptr0), shape=shape0)

    def _append_data(self, h5f, bmf, data, indices):
        # create refs to hdf5 data
        data0 = h5f['.']['data']
        indices0 = h5f['.']['indices']
        indptr0 = h5f['.']['indptr']
        shape0 = h5f['.']['shape']

        # update hdf5 data
        l0 = len(data0)
        l = len(data)
        data0.resize((l0+l,))
        data0[l0:] = data

        indices0.resize((l0+l,))
        indices0[l0:] = indices

        indptr0[:] = (0, l0+l)

        shape0[:] = (1, self._maxrows)

        # update map data
        bmf.map.update(indices)

    def put_data_blocks(self, blocksize, csr_blocks):
        with h5py.File(self._filepaths['data'], 'a') as h5f, MapFile(self._filepaths['map'], 'rw') as bmf:
            for bi, b in csr_blocks:
                self._append_data(h5f, bmf, b.data, b.indices + bi*blocksize)
        
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
            '!': lambda x: x.flip(0, MAXROWS-1)
        }
        unwrapper = lambda c: self._cols[self._colidx[c]].get_map() \
            if c in self._meta['cols'] else c
        self._rpn = rpn.ReversePolish(tokeniser, dispatcher, unwrapper)
        
    def _format(self, statement):
        return ' '.join(self._fmtpat.sub(' \\1 ', statement).split(' '))

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

    def exists(self):
        return os.path.isfile(self._filepaths['meta'])

    def create(self, cols):
        if self.exists():
            raise ValueError('database already exists')

        if len(cols) != len(set(cols)):
            raise ValueError('repeated column names')

        self._meta = {
            'cols': cols
        }

        with open(self._filepaths['meta'], 'w') as fp:
            toml.dump(self._meta, fp, sort_keys=True)
        
        self.attach()

    def attach(self):
        if not self.exists():
            raise ValueError('database does not exist')

        with open(self._filepaths['meta'], 'r') as fp:
            self._meta = toml.load(fp)

        self._colidx = {c:i for i,c in enumerate(self._meta['cols'])}
        self._cols = [SparseColumn(self._paths['/cols'], c) for c in self._meta['cols']]

    def find(self, statement):
        b = self._rpn.execute(statement)
        return list(b)

    def get_data(self, indices=None, cols=None):
        if cols is None:
            cols = self._meta['cols']
        if indices is None:
            return sparse.vstack(self._cols[c].get_data() for c in cols).T
        else:
            return sparse.vstack(self._cols[c].get_data()[1, indices] for c in cols).T

    def put_data_blocks(self, blocksize, csr_blocks):
        raise NotImplementedError
        for bi,blk in csr_blocks:
            blkc = blk.tocsc()
            for i,c in enumerate(blkc.T):
                self._cols[i].put_data_blocks([(bi,c)])
            
