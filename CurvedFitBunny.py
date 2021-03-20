#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vishelper import *
import tqdm
import osqp
from cubic_fitting import *


# In[2]:


_, _, sB, sT, V, F,_ = h5deser('../buildr/bunny.off.h5')


# In[3]:


sB,_ = igl.upsample(sB,F,2)
sT,_ = igl.upsample(sT,F,2)
V,F = igl.upsample(V,F,2)


# In[4]:


basis = bernstein_basis()
newV, newF, newE, SVI, SVJ, values_at_points, all_base, all_dir, all_global_ids = upsampled(sB, sT, F, basis, edge_globals(F), level=3)


# In[5]:


sp_dh_D_s = offset_from_ctrlpt_operator(values_at_points, all_global_ids, SVI, SVJ, all_dir, all_global_ids.max()+1, len(F))
g_tex = matcap_checkers(60,1)
updated_V = lambda s: update_V_from_ctrlpt(s, all_global_ids, values_at_points, all_base, all_dir, SVI)
V1 = updated_V(np.zeros(all_global_ids.max()+1))
sol1 = np.ones(all_global_ids.max()+1)*0.5


# ## Transplanting from Coarse

# In[6]:


with np.load('height_and_base.npz') as npl:
    c_height, c_base = npl['height'], npl['base']


# In[7]:


newdir = all_dir.reshape(-1,3)[SVI]
c_sol = scipy.sparse.linalg.spsolve(sp_dh_D_s.T@sp_dh_D_s,sp_dh_D_s.T@((newdir*c_height[:,None]).flatten()))


# In[27]:


vw0 = mp.Viewer(dict(width=800,height=800))
vw0.add_mesh(V1,newF)
vw0.add_edges(V1,newE)

vw1 = mp.Viewer(dict(width=800,height=800))
add_matcap_mesh(vw1,V1,newF,
                shading=dict(reflectivity=100.,flat=False,metalness=1.),
                texture_data=g_tex)

plt=None
plt = mp.Subplot(plt, vw0, s=[1,2,0])
plt = mp.Subplot(plt, vw1, s=[1,2,1])


# In[28]:


V1 = updated_V(c_sol)
newbase = (all_base.reshape(-1,3)[SVI].flatten())
for _ in range(50):
    laplacian = igl.cotmatrix(V1, newF).tocoo()
    massmat = igl.massmatrix(V1,newF).data
    
    print(((massmat**(-1))[:,None]*(laplacian@V1)**2).sum()/massmat.sum())
    #break
    lap3 = scipy.sparse.coo_matrix((np.concatenate([laplacian.data, laplacian.data, laplacian.data]),
    (np.concatenate([laplacian.row*3, laplacian.row*3+1, laplacian.row*3+2]),
    np.concatenate([laplacian.col*3, laplacian.col*3+1, laplacian.col*3+2]))))
    
    mass3 = np.tile(massmat,(3,1)).T.flatten() # aaabbbccc...
    A = scipy.sparse.diags(mass3**(-1/2))@lap3@sp_dh_D_s
    b = scipy.sparse.diags(mass3**(-1/2))@lap3@newbase
    
    prob = osqp.OSQP()
    prob.setup(A.T@A, A.T@b, 
               verbose=False)
    prob.warm_start(x=c_sol)
    prob_res = prob.solve()
    #print('From', (prob_res.info.obj_val*2+ b.T@b)/massmat.sum(), end=' ')

    c_sol = prob_res.x
    print(' ', (prob_res.info.obj_val*2 + b.T@b)/massmat.sum())

    V1 = updated_V(c_sol)
    vw0.reset()
    vw1.reset()
    vw0.add_mesh(V1,newF)
    vw0.add_edges(V1,newE)
    add_matcap_mesh(vw1,V1,newF,
                shading=dict(reflectivity=100.,flat=False,metalness=1.),
                texture_data=g_tex)

