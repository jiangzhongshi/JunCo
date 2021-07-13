#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vis_utils import *


# In[2]:


import sys
sys.path.append('../python')

from curve import fem_tabulator


# In[3]:


V,F = igl.read_triangle_mesh('/home/zhongshi/Aircraft_WL100_gmsh-auto.ply')


# In[5]:


A = igl.doublearea(V,F)


# In[ ]:


igl.face_components()


# In[4]:


import meshzoo


# In[12]:


iV,iF = meshzoo.sphere.icosa_sphere(1)


# In[85]:


get_ipython().system('ls ../buildr/Aircraft_WL20-edge_gmsh-auto.ply.h5_col0.h5*5e3.h5')


# In[80]:


mV,mF,cp = h5reader('../buildr/Aircraft_WL20-edge_gmsh-auto.ply.h5_col0.h5E1e1D5e3.h5','mV','mF','complete_cp')


# In[81]:


hV, hF, hE = highorder_sv(cp, level=3, order=2)


# In[82]:


igl.avg_edge_length(mV,mF)


# In[15]:


V,TC,_,F,FTC,_ = igl.read_obj('/home/zhongshi/AirCraftMark.obj')


# In[19]:


TT,TTi = igl.triangle_triangle_adjacency(FTC)


# In[21]:


edges = []
for i in range(len(TT)):
    for j in range(3):
        if TT[i,j] == -1:
            edges.append((F[i,j],F[i,j-2]))


# In[25]:


len(edges)


# In[33]:


uE = np.unique(np.sort(np.array(edges),axis=1),axis=0)


# In[35]:


with h5py.File('/home/zhongshi/Aircraft/feat_mark.h5', 'w') as fp:
    fp['E'] = uE


# In[49]:


mV,mF, meta_f, meta_i = h5reader('../buildr/AirCraftMark.obj.h5.init','ref.V','ref.F','meta_edges_flat', 'meta_edges_ind')


# In[56]:


mV,mF = h5reader('../buildr/AirCraftMark.obj.h5.init.h5_col0.h5','mV','mF')


# In[37]:


metas = np.split(meta_f, meta_i[1:-1])


# In[42]:


igl.doublearea(mV,mF).min()


# In[50]:


M = igl.massmatrix(mV,mF, igl.MASSMATRIX_TYPE_VORONOI)


# In[53]:


M.diagonal().min()


# In[ ]:




