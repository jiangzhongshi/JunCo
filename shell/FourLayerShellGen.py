#!/usr/bin/env python
# coding: utf-8

# In[2]:


from vis_utils import *


# In[2]:


import pymesh


# In[3]:


import glob
files = sorted(glob.glob('/home/zhongshi/data/0502_raw10k/*.h5'))


# In[29]:


def has_self_intersect(V,F):
    inter = pymesh.detect_self_intersection(pymesh.form_mesh(V,F)) 
    #print(len(inter))
    return len(inter) > 0


# In[35]:


def until_release(V,T, F):
    for _ in range(5):
        if not has_self_intersect(T,F):
            break
        T = (V+T)/2
    if _ == 10:
        return None
    return T


# In[40]:


import os,sys


# In[41]:


def process(f):
    f_out = '/home/zhongshi/public/shell_4layer/' + os.path.basename(f) + '.ply'
    V,B,T,F = h5reader(f, 'mV', 'mbase','mtop', 'mF')

    if has_self_intersect(V,F):
        return 1
    B,T = until_release(V,B,F),until_release(V,T,F)
    if B is None or T is None:
        return 2
    B0 = (B+V)/2
    T0 = (T+V)/2
    comb_V = np.vstack([B,B0,T0,T])
    igl.write_triangle_mesh(f_out, comb_V, F)
    return 0


# In[ ]:


import multiprocessing
pool = multiprocessing.Pool(100)
pool.map(process, files)


# In[57]:


V,F = igl.read_triangle_mesh('/home/zhongshi/public/shell_4layer/100070.stl.h5.ply')

plt = mplot(V,F)
for i in range(1,4):
    plt.add_edges(V[i*193:i*193+193],f2e(F))


# In[58]:


import pymesh


# In[59]:


plt = mplot(V,F)
for i in range(1,4):
    plt.add_edges(V[i*193:i*193+193],f2e(F))


# In[63]:


F_combine = np.vstack([F+i*193 for i in range(4)])


# In[4]:


def tetmesh_from_shell(base, top, F):
    tetra_splits = (np.array([0, 3, 4, 5, 1, 4, 2, 0, 2, 5, 0, 4]).reshape(-1, 4),
                    np.array([0, 3, 4, 5, 1, 4, 5, 0, 2, 5, 0, 1]).reshape(-1, 4))
    vnum = len(base)
    T = []
    for f in F:
        tet_c = tetra_splits[0] if f[1] > f[2] else tetra_splits[1]
        T.append((tet_c // 3)*vnum + f[tet_c % 3])
    return np.vstack([base, top]), np.vstack(T)


# In[3]:


V,F = igl.read_triangle_mesh('/home/zhongshi/public/shell_4layer/133991.stl.h5.ply')


# In[5]:


B1,B0,T0,T1 = np.split(V,4)


# In[16]:


Vtet,Ttet = tetmesh_from_shell(T0,T1, F)
print(igl.volume(Vtet,np.array([t for t  in Ttet if 0 not in t])).min())
Vtet,Ttet = tetmesh_from_shell(B1,B0, F)
print(igl.volume(Vtet,np.array([t for t  in Ttet if 0 not in t])).min())


# In[17]:


Ttet


# In[18]:


Vtet


# In[ ]:




