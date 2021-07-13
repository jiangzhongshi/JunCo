#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vis_utils import *


# In[2]:


import meshio


# In[14]:


import sys
sys.path.append('../python')


# In[15]:


from curve import fem_tabulator


# In[16]:


def convert_tri10(mF, cp, order):
    gmshcod = (fem_tabulator.codecs()['tri6'][2]*order).astype(np.int)
    autocod = fem_tabulator.tuple_gen(order=order,var_n=2)
    reorder = np.lexsort(
                np.array(autocod).T)[fem_tabulator.invert_permutation(np.lexsort(gmshcod.T))]
    #print(reorder)
    #print(autocod, gmshcod)
    assert np.all(np.array(autocod)[reorder] == gmshcod)

    tri10info = fem_tabulator.basis_info(order=order,nsd=2)

    def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i]*j]

    auto_cod_n = np.array([codec_to_n(c) for c in autocod])
    #print(auto_cod_n, np.sort(mF[:,auto_cod_n].reshape(-1,3)).shape)
    uniq_tup, tup_ind, tup_inv = np.unique(np.sort(mF[:,auto_cod_n].reshape(-1,order)), axis=0,
                                           return_index=True, return_inverse=True)
    #print(tup_inv.shape, mF.shape, cp.shape)
    #print(tup_inv[np.arange(len(cp)*6).reshape(-1,6)])
    m = meshio.Mesh(points=(tri10info['b2l']@cp).reshape(-1,3)[tup_ind], 
                    cells = [('triangle6', 
                              tup_inv[np.arange(len(cp)*6).reshape(-1,6)][:,reorder]
                            )])
    return m
#m = convert_tri10(mF, cp, order=2)


# In[55]:


V,F = igl.read_triangle_mesh('/home/zhongshi/Aircraft/Aircraft_WL20-edge_gmsh-auto.ply')
scale(V)
for e in ['1e2', '2e2', '1e1','5e2']:
    mV,mF,cp = h5reader(f'../buildr/Aircraft_WL20-edge_gmsh-auto.ply.h5_col0.h5E{e}D5e3.h5', 'mV', 'mF','complete_cp')
    m = convert_tri10(mF, cp*scale.b + scale.a, order=2)
    meshio.write(f'/home/zhongshi/Aircraft/WL20out/output_{e}.msh', m)


# In[82]:


metas = np.split(meta_f, meta_i[1:-1])


# In[ ]:




