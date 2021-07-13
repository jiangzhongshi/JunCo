#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import plotly.graph_objects as go
import tqdm
from vis_utils import *
import sys
sys.path.append('../python')
from curve import assembler, cubic_fitting
from curve import fem_tabulator


# In[149]:


mF,mV, mB, mT, refF, refV, meta_f, meta_i,track_f,track_s,cp = h5reader('../build_clang/cube_curve.h5',
                                                        'mF', 'mV', 'mbase', 'mtop', 'ref.F','ref.V',
                                    'meta_edges_flat','meta_edges_ind','track_flat','track_size','mF')
tracks = np.split(track_f.astype(np.int32).flatten(), np.cumsum(track_s.astype(np.int))[:-1])
metas = np.split(meta_f, meta_i[1:-1])


# In[10]:


level=3
usV, usF = igl.upsample(np.eye(3)[:,1:], np.arange(3)[None,:], level)
bnd0 = igl.boundary_loop(usF)
usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T


# ## Find correspondence employed to check.

# In[59]:


import seism

pc = seism.PrismCage('../build_clang/sculpt_curve.h5')
inpV,_ = h5reader('../build_clang/sculpt.h5','ref.V','ref.F')
flat_sv = np.einsum('fed,es->fsd',
                    np.einsum('fDd,eD->fed',mV[mF], np.array(tri10['codec']))/3,
                    bas_val)
pc_fid, pc_uv = pc.transfer(refV, refF,flat_sv.reshape(-1,3))

sampled_pos = np.einsum('fed,fe->fd',inpV[refF[pc_fid.astype(np.int)]], np.hstack([1-pc_uv.sum(axis=1, keepdims=True), pc_uv]))


# In[60]:


trans_uv = np.einsum('fed,fe->fd',inpV[refF[pc_fid.astype(np.int)], :2], np.hstack([1-pc_uv.sum(axis=1, keepdims=True), pc_uv]))


# In[15]:


inpV,inpF = h5reader('../build_clang/sculpt.h5','ref.V','ref.F')

