#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas
import pymesh
from vis_utils import *


# In[15]:


def scale(V):
    scale.a = (V.min(axis=0))
    V = V - scale.a
    scale.b = V.max()
    return V/scale.b 
use_scale = lambda V: (V-scale.a)/scale.b


# In[11]:


V0, UV0,_,F0,TF0, _ = igl.read_obj('/home/zhongshi/Workspace/nutshell_data/thai/thai_optcuts.obj')
V1, UV1,_,F1,TF1, _ = igl.read_obj('/home/zhongshi/Workspace/nutshell_data/thai/thai_optcuts-decimated_to_2000_vertices.obj')


# In[17]:


igl.write_obj('/home/zhongshi/Workspace/nutshell_data/thai/optcuts.obj', use_scale(V1),F1)


# In[ ]:


def scale(V): # the legacy scale function, in order to use old camera etc.
    scale.a = (V.max(axis=0)+V.min(axis=0))/2
    V = V - scale.ax
    scale.b = V.max()
    return V/scale.b 
use_scale = lambda V: (V-scale.a)/scale.b


# In[ ]:





# In[3]:


# V0,F0 = igl.upsample(V0,F0,4)
# UV0,TF0 = igl.upsample(UV0,TF0,4)
V1,F1 = igl.upsample(V1,F1,6)
UV1,TF1 = igl.upsample(UV1,TF1,6)


# In[4]:


d1 = igl.exact_geodesic(V0,F0.astype('i'), vs=np.array([603], dtype='i'), vt=np.arange(len(V0), dtype='i'))


# In[5]:


mesh = pymesh.form_mesh(UV0,TF0)
_, fid, pts = pymesh.distance_to_mesh(mesh, UV1)
A, B, C = [np.hstack([UV0[TF0[fid, i]], np.zeros((len(fid),1))]) for i in range(3)]


# In[6]:


qUV = igl.barycentric_coordinates_tri(np.hstack([pts, np.zeros((len(fid),1))]),A,B,C)


# In[7]:


pd1 = np.sum(d1[F0[fid]]*qUV,axis=1)


# In[8]:


def index_corres_uv(v_num, F, TF):
    cor = -np.ones(v_num,dtype='i')
    for f, tf in zip(F,TF):
        for j in range(3):
            cor[f[j]] = tf[j]
    return cor
invcor1 = index_corres_uv(len(V1), F1, TF1)


# In[ ]:




