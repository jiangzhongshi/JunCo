#!/usr/bin/env python
# coding: utf-8

# In[46]:


from vis_utils import *


# In[47]:


import pandas as pd
import plotly.express as px


# In[48]:


energy_b, lagr, p4T = h5reader('../buildr/before.h5','energy', 'lagr', 'p4T')
energy_a,_ = h5reader('../buildr/after.h5','energy','lagr')


# In[49]:


verts_on_bnd = set(np.unique(igl.boundary_facets(p4T[:,:4])))

interior_mark = np.array([len(set(t)&verts_on_bnd) == 0 for t in p4T])

df = pd.DataFrame(list(zip(energy_b, energy_a, interior_mark,*lagr[p4T[:,:4]].mean(axis=1).T)),
            columns =['E_b','E_a', 'in', 'x','y','z'])


# In[50]:


px.scatter(df, x='x',y='E_a')


# # Are we computing mips correctly?

# In[43]:


def mips3d(V,T):
    Jacs = V[T[:,1:]]-V[T[:,:1]]
    frob = np.sum(Jacs**2,axis=1).sum(axis=1)
    invJ = np.linalg.inv(Jacs)
    invfrob = np.sum(invJ**2,axis=1).sum(axis=1)
    return frob*invfrob


# In[2]:


import meshio


# In[3]:


import sys
sys.path.append('../python')
from curve import fem_tabulator
gmsh_cod = (fem_tabulator.codecs()['tetra35'][-1]*4).astype(np.int)
auto_cod = np.array(fem_tabulator.tuple_gen(order=4, var_n=3))
reorder = np.lexsort(
        (auto_cod).T)[fem_tabulator.invert_permutation(np.lexsort(gmsh_cod.T))]
assert np.all((auto_cod)[reorder]== gmsh_cod)


# In[4]:


energy_a, lagr, p4T = h5reader('../buildr/after.h5','energy','lagr', 'cells')
energy_a.max()


# In[8]:


import meshio
meshio.write('/home/zhongshi/public/curved/block.msh',
            meshio.Mesh(points=lagr, cells=[('tetra35', p4T[:,reorder])]),
             file_format='gmsh22', binary=False)


# In[6]:


meshio.write('/home/zhongshi/public/curved/block_opt1.msh',
            meshio.Mesh(points=lagr, cells=[('tetra35', p4T[451:452,reorder])]),
             file_format='gmsh22', binary=False)


# In[28]:


import meshio
m = meshio.read('/home/zhongshi/public/curved/block.msh')


# In[32]:


p4T = m.cells[0][1]


# In[38]:


TT,TTi = igl.tet_tet_adjacency(p4T[:,:4])


# In[37]:


np.intersect1d(p4T[0], p4T[177])


# # Generate and stitch the mesh

# In[ ]:


mV, mF, mB, top,cp = h5reader('/home/zhongshi/data/1222_cumin/block_input_tri.obj.h5',
                              'mV','mF','mbase','mtop',
                                     'complete_cp')


# In[ ]:


from curve import surface_to_curved_layer


# In[ ]:


bern2elevlag, _ = h5reader(
    '../python/curve/data/tri_o3_lv3.h5', 'bern2elevlag', 'bern')


# In[5]:


tetgen = surface_to_curved_layer.conforming_tetegen(mB, mF)


# In[ ]:


mesh = surface_to_curved_layer.surface_to_curved_layer(
                     mB, mF, cp, bern2elevlag, tetgen)


# In[ ]:


p4T = mesh.cells[0][1]
TT,TTi = igl.tet_tet_adjacency(p4T[:,:4])
for i,t in enumerate(TT):
    for j in t:
        if j != -1:
            assert(len(np.intersect1d(p4T[i], p4T[j]))>=15)


# In[ ]:


import meshio
meshio.write('/home/zhongshi/public/curved/block_fine.msh',
                 mesh,
                 file_format='gmsh22', binary=False)


# In[ ]:


from curve import fem_tabulator
gmsh_cod = (fem_tabulator.codecs()['tetra35'][-1]*4).astype(np.int)
auto_cod = np.array(fem_tabulator.tuple_gen(order=4, var_n=3))
reorder = np.lexsort(
        (auto_cod).T)[fem_tabulator.invert_permutation(np.lexsort(gmsh_cod.T))]
assert np.all((auto_cod)[reorder]== gmsh_cod)


# In[ ]:


with h5py.File('../buildr/block_fine_autocod.h5','w') as fp:
    fp['lagr'] = mesh.points
    fp['cells'] = p4T[:,fem_tabulator.invert_permutation(reorder)]


# In[ ]:


energy_a, lagr, p4T = h5reader('../buildr/after.h5','energy','lagr', 'cells')
energy_a.max()


# In[ ]:


meshio.write('/home/zhongshi/public/curved/block_fine_opt.msh',
            meshio.Mesh(points=lagr, cells=[('tetra35', p4T[:,reorder])]),
             file_format='gmsh22', binary=False)

