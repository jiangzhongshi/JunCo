#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vishelper import *
import tqdm
import osqp
from cubic_fitting import *


# In[2]:


def simple_spherical_intersect(sB, sD):
    dd = np.sum(sD*sD, axis=1)
    db = 2*np.sum(sD*sB, axis=1)
    bb = np.sum(sB*sB, axis=1) - 1
    ray_steps = (-db + np.sqrt(db**2 - 4*dd*bb))/(2*dd)
    return ray_steps


# In[3]:


basis = bernstein_basis()
g_tex = matcap_checkers(60,1)
updated_V = lambda s: update_V_from_ctrlpt(s, all_global_ids, values_at_points, all_base, all_dir, SVI)


# In[59]:


V,F = igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/cube.obj')
for level in range(4):
    V,F = igl.upsample(V,F,level)

    sB = V/np.sqrt(3)
    sT = V*np.sqrt(3)

    newV, newF, newE, SVI, SVJ, values_at_points, all_base, all_dir, all_global_ids = upsampled(sB, sT, F, basis, edge_globals(F), level=6-level)

    sp_dh_D_s = offset_from_ctrlpt_operator(values_at_points, all_global_ids, SVI, SVJ, all_dir, all_global_ids.max()+1, len(F))

    steps = simple_spherical_intersect(all_base.reshape(-1,3)[SVI], all_dir.reshape(-1,3)[SVI])

    prob = osqp.OSQP()
    prob.setup(sp_dh_D_s.T@sp_dh_D_s,-sp_dh_D_s.T@(steps[:,None]*all_dir.reshape(-1,3)[SVI]).flatten(),
    #            A = scipy.sparse.eye(sp_dh_D_s.shape[1]).tocsc(), 
    #            l = np.concatenate([np.ones(8)*0.2,
    #                                np.zeros(sp_dh_D_s.shape[1]-8)]),
    #            u = np.concatenate([np.ones(8)*0.2,
    #                                np.ones(sp_dh_D_s.shape[1]-8)]),
               verbose=False)
    prob_res = prob.solve()
    sol0 = prob_res.x

    print(np.linalg.norm(updated_V(sol0) - (steps[:,None]*all_dir.reshape(-1,3)[SVI] +all_base.reshape(-1,3)[SVI])))


# In[9]:


V,F = igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/cube.obj')
V,F = igl.upsample(V[:3],F[:1],6)
usV, usF = igl.upsample(
        np.array([[0, 0.], [1, 0], [0, 1]]), np.array([[0, 1, 2]]), 6)
sB = V
sT = V*np.sqrt(3)


# In[10]:


steps = simple_spherical_intersect(sB,sT-sB)


# In[70]:


u,v = usV[:,0],usV[:,1]
steps = u**2 *v + 2*u*v**2 + 2*u*v + u 


# In[71]:


A = np.array([f(usV[:,0].flatten(), usV[:,1].flatten()) for f in basis]).T


# In[72]:


sol = np.linalg.solve(A.T@A, A.T@steps)
vw0 = mp.Viewer(dict(width=600,height=600))
vw0.add_mesh(usV,usF)
vw0.add_mesh(np.hstack([usV, steps[:,None]]),usF, c = np.array([0.8,0.8,0.8]),shading=dict(wireframe=True))
vw0.add_points(np.hstack([usV, (A@sol)[:,None]]))

vw1 = mp.Viewer(dict(width=600,height=600))
vw1.add_mesh(usV,usF)
vw1.add_mesh(np.hstack([usV, (A@sol)[:,None]]),usF, shading=dict(wireframe=True))
vw1.add_points(np.hstack([usV, steps[:,None]]))


vw2 = mp.Viewer(dict(width=600,height=600))
vw2.add_mesh(usV,usF)
vw2.add_mesh(np.hstack([usV, steps[:,None]]),usF, c = np.array([0.8,0.8,0.8]),shading=dict(wireframe=True))
vw2.add_mesh(np.hstack([usV, (A@sol)[:,None]]),usF, shading=dict(wireframe=True))


# In[76]:


np.linalg.norm(steps - A@sol)


# In[16]:


sol1 = sol0
V1 = updated_V(sol1)

newbase = (all_base.reshape(-1,3)[SVI].flatten())
for _ in range(50):
    laplacian = igl.cotmatrix(V1, newF).tocoo()
    massmat = igl.massmatrix(V1,newF).data
    
    print('True', (massmat[:,None]**(-1)*(laplacian@V1)**2).sum(), end=' ')
    if np.any(np.isnan(laplacian.data)):
        print('Warning')
        break
    lap3 = scipy.sparse.coo_matrix((np.concatenate([laplacian.data, laplacian.data, laplacian.data]),
    (np.concatenate([laplacian.row*3, laplacian.row*3+1, laplacian.row*3+2]),
    np.concatenate([laplacian.col*3, laplacian.col*3+1, laplacian.col*3+2]))))
    
    mass3 = np.tile(massmat,(3,1)).T.flatten() # aaabbbccc...
    A = scipy.sparse.diags(mass3**(-1/2))@lap3@sp_dh_D_s
    b = scipy.sparse.diags(mass3**(-1/2))@lap3@newbase
    
    prob = osqp.OSQP()
    prob.setup(A.T@A, A.T@b, 
               A=scipy.sparse.eye(A.shape[1]).tocsc(), l = np.concatenate([np.ones(8)*0.2,np.zeros(A.shape[1]-8)]), u = np.concatenate([np.ones(8)*0.2,np.ones(A.shape[1]-8)]),
               verbose=False)
    prob.warm_start(x=sol1)
    prob_res = prob.solve()
    sol1 = prob_res.x
    print((prob_res.info.obj_val*2 + b.T@b))
    #iter_count = iter_count+1
    
    if np.any(np.isnan(sol1)):
        print('solwarn')
        break
    
    V1 = updated_V(sol1)
#     vw0.reset()
#     vw0.add_mesh(V1,newF,shading=dict(flat=False))
#     vw0.add_edges(V1,newE)
#     vw1.reset()
#     add_matcap_mesh(vw1,V1,newF,
#                     shading=dict(reflectivity=100.,flat=False,metalness=1.),
#                     texture_data=g_tex)


# In[19]:


vw0 = mp.Viewer(dict(width=1600,height=1600))
vw1 = mp.Viewer(dict(width=1600,height=1600))
vw0.reset()
vw1.reset()
vw0.add_mesh(V1,newF,shading=dict(flat=False))
vw0.add_edges(V1,newE)
add_matcap_mesh(vw1,V1,newF,
                shading=dict(reflectivity=100.,flat=False,metalness=1.),
                texture_data=g_tex)


# In[ ]:




