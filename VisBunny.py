#!/usr/bin/env python
# coding: utf-8

# In[4]:


from vis_utils import *


# In[1]:


import meshio


# In[35]:


refVF_f, refVF_i = igl.vertex_triangle_adjacency(refF, len(refV))
refVF = np.split(refVF_f, refVF_i[1:-1])
assert len(refVF) == len(refV)

metas = np.split(meta_f, meta_i[1:-1])
vids = set(np.concatenate([m[2:] for m in metas if m[2] == 0]))
mE = np.array([(a,b)for m in metas if m[2] == 0  for a,b in zip(m[3:-1],m[4:]) ])
dealing_faces = set(np.concatenate([refVF[v] for v in vids]))

def vv2fe(a,b, F, VF): # temporary. gives the arbitrary of two answers
    for f in VF[a]:
        if b in F[f]:
            return f, (np.where(F[f]==b)[0][0])

mFE = [vv2fe(a,b, refF,refVF) for a,b in mE]

V, F, vmap, vinv = igl.remove_unreferenced(inpV,refF[list(dealing_faces)])

Evv = vmap[mE]

VF_f, VF_i = igl.vertex_triangle_adjacency(F, len(V))
VF = np.split(VF_f, VF_i[1:-1])
TT,TTi = igl.triangle_triangle_adjacency(F)

Emark = -1*np.ones_like(TT)

for a,b in Evv:
    f,e = vv2fe(a,b,F,VF)
    Emark[f,e] = 0
    Emark[TT[f,e],TTi[f,e]] = 0

angles = igl.internal_angles(V,F)[:,[2,0,1]]

np.where(np.logical_and(angles>2, Emark>=0 ))

p = mp.plot(V,F[[64,  65,  80,  82,  83,  88,  89,  91,  93,  94,  95,  96,  98,
         99, 100]],wireframe=True)
p.add_edges(V,vmap[mE], shading=dict(point_size=0.1))

