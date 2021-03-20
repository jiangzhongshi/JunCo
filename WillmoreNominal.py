#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vis_utils import *


# In[2]:


V,F,fe = h5reader('../build_profile/block_collapse.h5','ref.V','ref.F','feat_edges')


# In[38]:


V,F= igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/xcylinder.obj')
print(F.shape)


# In[48]:


import meshzoo


# In[98]:


V, F = meshzoo.tube(length=5.0, radius=1.0, n=10)


# In[99]:


V2,F2 = igl.upsample(V,F,2)
print(F2.shape)


# In[100]:


V2_0,F2 = igl.upsample(V,F,2)


# In[101]:


def repeat_diags(laplacian):
    return scipy.sparse.coo_matrix((np.concatenate([laplacian.data, laplacian.data, laplacian.data]),
                                            (np.concatenate([laplacian.row*3, laplacian.row*3+1, laplacian.row*3+2]),
                                             np.concatenate([laplacian.col*3, laplacian.col*3+1, laplacian.col*3+2]))))


# In[102]:


L = - igl.cotmatrix(V2,F2)
M = igl.massmatrix(V2,F2)
P = L@scipy.sparse.diags(M.data**(-1))@L
energy_history = [V2.flatten()[None,:]@repeat_diags(P.tocoo())@V2.flatten()/4]


# In[103]:


bnd = np.unique(igl.boundary_facets(F2).flatten())
L = - igl.cotmatrix(V2,F2)
M = igl.massmatrix(V2,F2)
P = L@scipy.sparse.diags(M.data**(-1))@L
P = P.T@P
import osqp
prob = osqp.OSQP()
fixed_index = np.unique(np.concatenate([bnd, np.arange(len(V))]))


# In[104]:


extend_index = (np.tile(fixed_index[:,None],(1,3))*3 + np.arange(3)).flatten()


# In[105]:


A = scipy.sparse.diags(extend_index)
prob.setup(P = repeat_diags(P.tocoo()).tocsc(), q=np.zeros(V2.size), 
           A = scipy.sparse.eye(V2.size).tocsc()[extend_index],
           l = V2.flatten()[extend_index], u=V2.flatten()[extend_index])
result = prob.solve()
V2 = result.x.reshape(-1,3)
energy_history.append(V2.flatten()[None,:]@repeat_diags(P.tocoo())@V2.flatten()/4)
print(energy_history)


# In[ ]:




