#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py


# In[3]:


import sys
sys.path.append('../python')
from curve import fem_tabulator


# In[4]:


import igl
import numpy as np
import itertools


# In[5]:


codec_to_n = lambda co: [k for i,j in enumerate(co) for k in [i]*j]


# In[6]:


level=3
usV, usF = igl.upsample(
    np.array([[0, 0.], [1, 0], [0, 1]]), np.array([[0, 1, 2]]), level)
bnd0 = igl.boundary_loop(usF)
usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
u, v = usV[:, :1], usV[:, 1:]


# In[36]:


for order in range(1,3):
    tri_o = fem_tabulator.basis_info(order=order, nsd=2, derivative=True)
    basis = tri_o['basis']
    tri_o1 = fem_tabulator.basis_info(order=order+1, nsd=2, derivative=False)
    cod_o1 = tri_o1['codec']

    values_at_nodes = np.hstack(basis(u, v)).T
    sample_deri = np.array([[f(*p) for p in usV] for f in tri_o['basis_d']])
    _, cod_o4_u, cod_o4_v = np.array(tri_o['codec']).T/(order)
    _, cod_o5_u, cod_o5_v = np.array(cod_o1).T/(order+1)
    assert(cod_o5_u.max() == 1)
    with h5py.File(f'../python/curve/data/tri_o{order}_lv{level}.h5','w') as f:
        f['bern'] = values_at_nodes
        f['bern2elevlag'] = np.vstack(basis(cod_o5_u,cod_o5_v))
        f['bern2lag'] = np.vstack(basis(cod_o4_u,cod_o4_v))
        f['deri_u'] = sample_deri[0]
        f['deri_v'] = sample_deri[1]


# In[7]:


N=5
sample_pts = np.asarray(list(filter(lambda x: sum(x) == N, itertools.product(range(N+1), repeat=4))))[:,1:]/N


# In[41]:


for order in range(2,5):
    tet_o = fem_tabulator.basis_info(order=order+1, nsd=3, derivative=True)
    bern_from_lag = tet_o['l2b']
    with h5py.File(f'../python/curve/data/p{order+1}_quniform5_dxyz.h5','w') as f: 
        f['dxyz'] = np.array([[f(*p) for p in sample_pts] for f in tet_o['basis_d']])@bern_from_lag


# In[ ]:




