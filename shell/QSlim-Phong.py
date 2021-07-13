#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas
import pymesh
from vis_utils import *


# In[3]:


refV,refF, B,T,V,F, _ = h5deser('../buildr/thai_statue.obj.h5')


# In[7]:


V,F = igl.read_triangle_mesh('../buildr/thai_statue.obj.h5.ply')


# In[110]:


d1 = igl.exact_geodesic(refV,refF.astype('i'), vs=np.array([603], dtype='i'), vt=np.arange(len(refV), dtype='i'))
iV1, iE1 = igl.isolines(refV,refF, d1*10, 30)


# In[129]:


igl.write_triangle_mesh('/home/zhongshi/Workspace/nutshell_data/thai/ref.obj', refV,refF)
igl.write_triangle_mesh('/home/zhongshi/Workspace/nutshell_data/thai/shell.obj', V,F)


# In[128]:


igl.write_triangle_mesh('/home/zhongshi/Workspace/nutshell_data/thai/ref_wires.obj', iV1, np.hstack([iE1, np.ones((len(iE1),1)).astype(np.int)]))


# ## Phong

# In[23]:


_,V1,F1,_,_ = igl.qslim(refV,refF,2000)
pymesh.detect_self_intersection(pymesh.form_mesh(V1,F1))


# In[24]:


import sys,os
sys.path.append('../python/')
import seism


# In[25]:


VN1 = igl.per_vertex_normals(V1,F1)


# In[26]:


usV, usF = igl.upsample(
        np.array([[0, 0.], [1, 0], [0, 1]]), np.array([[0, 1, 2]]), 6)


# In[27]:


u, v = usV[:, :1], usV[:, 1:]
V_combine = np.einsum('fjk,bj->fbk', V1[F1], np.hstack([1-u-v, u, v])).reshape(-1,3)
F_combine = np.vstack([usF+len(u)*i for i in range(len(F1))])
N_combine = np.einsum('fjk,bj->fbk', VN1[F1], np.hstack([1-u-v, u, v])).reshape(-1,3)


# In[28]:


SV, SVI, SVJ, _ = igl.remove_duplicate_vertices(V_combine,F_combine, 1e-10)
SF = SVJ[F_combine]
SN = N_combine[SVI]


# In[29]:


import tqdm
aabb = seism.AABB(refV,refF)
queries = -np.ones((len(SV),3))
for i,(v,n) in enumerate(zip(tqdm.tqdm(SV), SN)):
    side1 = aabb.segment_hit(v, v+n, True)
    side2 = aabb.segment_hit(v, v-n, True)
    if side1[0] == -1 or side1[-1] > side2[-1]:
        side1 = side2
    queries[i] = side1[:-1]


# In[39]:


len(queries)


# In[38]:


np.count_nonzero(queries[:,0]==-1)


# In[119]:


trans_color1 = np.zeros(len(queries))
for i,(f,u,v) in enumerate(tqdm.tqdm(queries)):
    trans_color1[i] = d1[refF[int(f)]].dot(np.array([1-u-v,u,v]))


# In[133]:


piV1, piE1 = igl.isolines(SV,SF, trans_color1, 30)


# In[121]:


igl.write_triangle_mesh('/home/zhongshi/Workspace/nutshell_data/thai/qslim_wires.obj', piV1, np.hstack([piE1, np.ones((len(piE1),1)).astype(np.int)]))


# In[130]:


igl.write_triangle_mesh('/home/zhongshi/Workspace/nutshell_data/thai/qslim.obj', V1,F1)


# In[134]:


plt = mplot(V1,F1,wireframe=False)
plt.add_edges(piV1,piE1)


# ## Shell

# In[122]:


fid, qUV = h5reader('../buildr/thai_statue.obj.up4.h5','fid','qUV')


# In[123]:


uV,uF = igl.upsample(V,F,6)


# In[125]:


u, v = qUV[:,:1], qUV[:,1:]
query_d1 = np.sum(d1[refF[fid]]*np.hstack([1-u-v, u, v]),axis=1)


# In[126]:


piV1, piE1 = igl.isolines(uV,uF, query_d1*10, 30)


# In[127]:


igl.write_triangle_mesh('/home/zhongshi/Workspace/nutshell_data/thai/shell_wires.obj', piV1, np.hstack([piE1, np.ones((len(piE1),1)).astype(np.int)]))


# In[ ]:




