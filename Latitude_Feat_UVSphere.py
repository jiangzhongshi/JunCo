#!/usr/bin/env python
# coding: utf-8

# In[2]:


from vis_utils import *


# # Generation Part

# In[3]:


V, F= igl.read_triangle_mesh('../tests/data/bichon/uv-sphere.obj')


# `V[:,1]` is the direction from pole to pole, range from -1. to 1. 
# There are 64 rings to consider, equator is 32.

# In[4]:


uniq, inve = np.unique(np.round(V[:,1]*1e5),return_inverse=True)


# In[5]:


E = igl.edges(F)


# In[6]:


all_E = [np.array([e for e in E if inve[e[0]]==i and inve[e[1]]==i]) for i in range(64)]


# In[7]:


# sample edges
for s in range(1,20):
    ch = np.random.choice(range(30,64), s,replace=False)
    Ec = np.vstack([all_E[i] for i in ch])
    with h5py.File(f'../tests/data/bichon/uv-feat.up{s}.h5','w') as fp:
        fp['E'] = Ec


# # Inspect Part

# In[3]:


import glob
glob.glob('../tests/data/bichon/uv*obj??.h5')


# In[4]:


picks = [2,5,10, 15,19]


# In[41]:


for f in picks:
    #mp.plot(mV,mF,wireframe=True)
    cp,mF = h5reader(f'../tests/data/bichon/uv-sphere.obj{f}.h5', 'complete_cp','mF')
    hV,hF,hE = highorder_sv(cp,level=4)
    igl.write_triangle_mesh(f'../tests/data/bichon/uv-feat/sp{f}-tri.obj',hV.reshape(-1,3), hF)
    write_obj_lines(f'../tests/data/bichon/uv-feat/sp{f}-edge.obj', hV.reshape(-1,3), hE)


# In[97]:


def vv2fe(a,b, F, VF): # temporary. gives the arbitrary of two answers
    for f in VF[a]:
        for e in range(3):
            if a == F[f][e] and b == F[f][e-2]:
                return f,e
def local_upsample(level:int):
    usV, usF = igl.upsample(np.eye(3)[:,1:], np.arange(3)[None,:], level)
    bnd0 = igl.boundary_loop(usF)
    usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
    return usV, usF, np.array_split(usE,3)


# In[98]:


for f in picks:
    mV, cp,mF,meta_i, meta_f = h5reader(f'../tests/data/bichon/uv-sphere.obj{f}.h5', 'mV','complete_cp','mF', 'meta_edges_ind', 'meta_edges_flat')
    refVF_f, refVF_i = igl.vertex_triangle_adjacency(mF, len(mF))
    refVF = np.split(refVF_f, refVF_i[1:-1])
    metas = np.split(meta_f, meta_i[1:-1])
    fe_list = np.array([vv2fe(m[0],m[1],mF,refVF) for m in metas])
    
    hV,hF,hE = highorder_sv(cp,level=5)
    uV,uF,uE =local_upsample(5)
    total_e = np.vstack([f*len(uV) + uE[e] for f,e in fe_list])
    write_obj_lines(f'../tests/data/bichon/uv-feat/feat{f}.obj', hV.reshape(-1,3), total_e)


# In[39]:





# In[50]:





# In[51]:





# In[86]:





# In[ ]:




