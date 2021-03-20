#!/usr/bin/env python
# coding: utf-8

# In[111]:


from vis_utils import *


# In[112]:


import glob


# In[113]:


import os


# In[114]:


basepath = '/home/zhongshi/data/abc_chunk0/'


# In[115]:


h5files = glob.glob(basepath + 'feat_h5/*.yml.h5')


# In[116]:


stranges = []
for f in tqdm.tqdm(h5files):
    name = os.path.basename(f).split('.yml')[0]
    E = h5reader(f'{basepath}/feat_h5/{name}.yml.h5','E')[0]
    objname = name.replace('features','trimesh')
    V,F = igl.read_triangle_mesh(f'/home/zhongshi/data/abc_chunk0/filter_obj/{objname}.obj')
    V = scale(V)
    FN = igl.per_face_normals(V,F,np.ones(3))
    TT,TTi = igl.triangle_triangle_adjacency(F)
    di_angles = (FN[TT]*FN[:,None,:]).sum(axis=2)
    dE =np.array([(F[f,e], F[f,e-2]) for f,e in zip(*np.where(di_angles < 0.))])
    if len({tuple(sorted(e)) for e in dE} - {tuple(sorted(e)) for e in E}) != 0:
        stranges.append(f)


# In[122]:


stranges


# In[119]:


len(stranges)/ len(h5files)

