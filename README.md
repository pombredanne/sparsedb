# sparsedb

silly non-database for storing and querying large sparse matrices
based on sparsity patterns (wip)

## usage

```python
from scipy import sparse
from sparsedb import SparseDB

# set database information
db_path = '/path/to/db'
db_name = 'some_db'
db_cols = ['col%d' % i for i in range(0,4)]

# generate data to be loaded
blocksize = 8 
B0 = sparse.csr_matrix([(0.,)+p for p in product([0,1],repeat=3)])
B1 = sparse.csr_matrix([(1.,)+p for p in product([0,1],repeat=3)])
blocks = [(0, B0), (1, B1)]

# create and connect to database
sdb = SparseDB(db_path, db_name)
if not sdb.exists():
    sdb.create(db_cols)
    sdb.put_data_blocks(blocksize=blocksize, csr_blocks=blocks)
else:
    sdb.attach()

# find rows based on arbitrary sparsity patterns
# query language uses reverse polish notation with logical ops & | ^ !
idx = sdb.find('col1 col2 col3 & |')

# get data corresponding to found rows
dta = sdb.get_data(cols = ['col0', 'col1', 'col2', 'col3'], indices=idx)
print(dta.todense())
```
