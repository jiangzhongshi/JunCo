#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy.optimize import linprog
import meshplot as mp
import numpy as np
import igl
import torch
from plotly import graph_objects as go

import sys


# In[ ]:


def subdivide_prism(tri, vnum):
    v0 = np.argmin(tri)
    tri = np.roll(tri, -v0) # put smallest first
    prism = np.concatenate([tri, tri+vnum])
    if tri[1] > tri[2]:
        return prism[subdivide_prism.type1]
    else:
        return prism[subdivide_prism.type2]
subdivide_prism.type1 =  np.array([[0,1,2,4], # bottom
               [0,4,2,5], # middle
               [0,4,5,3]]) # top
subdivide_prism.type2 = np.array([[0,1,2,5],
                        [0,1,5,4],
                        [0,4,5,3]])
def volume(V,T):
    all_std = np.einsum('ab,cbd->cad',volume.subtract_row0, V[T])
    return np.linalg.det(all_std)/6
volume.subtract_row0 = np.hstack([-np.ones((3,1)), np.eye(3)])


# In[ ]:


import tqdm
steps = np.ones(V.shape[0])*1e-4

#dynembree = seism.EmbreeIntersector()
count = 0

VT = V+ steps[:,None]*VN
assert not seism.cgal.polyhedron_self_intersect(VT, F)

aabb = seism.AABB(V,F)

total_v = V.shape[0]
fullV = np.vstack([V, VT])

pbar = tqdm.trange(total_v)
for vid in pbar:
    nbF = VF_list[vid]
    nbT = T[[3*f+i for f in nbF for i in range(3)]]
    alpha = 1e-2

    VT0 = VT[vid].copy()

    for _ in range(10):
        
        fullV[total_v + vid] += alpha * VN[vid]
        minvol = np.min(volume(fullV, nbT))

        if minvol > 0:
            if not any(aabb.triangle_query(fullV[F[f]+total_v]) for f in nbF):
                break
        alpha /=2
        fullV[total_v + vid] = VT0 # recover
    else:
        continue
    #assert not seism.cgal.polyhedron_self_intersect(fullV[total_v:], F) # predicate, revert.
#         fullV[total_v + vid] = VT0 # recover
#         count += 1
#         pbar.set_description(f'rev:{count}')


# In[ ]:


pl = mp.plot(V + 0.2*(fullV[total_v:]-V), F, return_plot=True)
pl.add_mesh(V,F,c=np.ones((V.shape[0],3)))


# In[ ]:


mp.plot(np.vstack([V,end_tip]), T)


# In[ ]:


import tqdm
steps = np.ones(V.shape[0])*1e-4

dynembree = seism.EmbreeIntersector()
count = 0

VT = V+ steps[:,None]*VN
assert not seism.cgal.polyhedron_self_intersect(VT, F)
dynembree.init(VT,F,True)

total_v = V.shape[0]
fullV = np.vstack([V, VT])

for vid in tqdm.trange(total_v):
    nbF = VF_list[vid]
    nbT = T[[3*f+i for f in nbF for i in range(3)]]
    alpha = 1e-1

    VT0 = VT[vid].copy()

    for _ in range(10):
        
        fullV[total_v + vid] += alpha * VN[vid]
        minvol = np.min(volume(fullV, nbT))

        if minvol > 0:
            hf, hu, hv = dynembree.intersectBeam(fullV[total_v + vid], -VN[vid]) # from trial backwards normal
            assert hf != -1
            if hf in nbF: # first hit is within one ring, nothing in between
                break
        alpha /=2
        fullV[total_v + vid] = VT0 # recover
    else:
        continue
    if seism.cgal.polyhedron_self_intersect(fullV[total_v:], F): # predicate, revert.
        fullV[total_v + vid] = VT0 # recover
        count += 1
    else:
        dynembree.deformVertex(vid, fullV[total_v + vid])


# In[ ]:


tetra_grads = np.zeros((3*F.shape[0], 3,3))
for f in range(F.shape[0]):
    local_conn = T[3*f:3*f+3]
    ref_geom = (np.vstack([V[local_conn[0][:-1]], V[local_conn[0][:-1]] + 1e-2*FN[f]]))
    if local_conn[0][2] == local_conn[1][2]:
        ref_conn = subdivide_prism.type1
    else:
        ref_conn = subdivide_prism.type2
        
    for i,rc in enumerate(ref_conn):
        ref_tet = ref_geom[rc]
        tetra_grads[3*f + i] = (np.linalg.inv(ref_tet[1:]-ref_tet[0]))


# In[ ]:


TG = torch.from_numpy(tetra_grads)
thV = torch.from_numpy(V)
thVN = torch.from_numpy(VN)


# In[ ]:


step_len = (torch.ones(V.shape[0])*1e-4).double().requires_grad_()


# In[ ]:


EndTip = torch.from_numpy(end_tip)
EndTip.requires_grad_()


# In[ ]:


def tetra_vol(step_len):
    EndTip = thV + step_len[:,None]*thVN
    cur_V = torch.cat([thV, EndTip])
    all_std = torch.einsum('ab,cbd->cad', torch.from_numpy(np.hstack([-np.ones((3,1)), np.eye(3)])), cur_V[T])
    return torch.det(all_std).sum()


# In[ ]:


step_len.grad.data.zero_()
tetra_vol(step_len).backward()


# In[ ]:


with torch.no_grad():
    print('E0', tetra_vol(step_len).item())
    for eps in [1e-9,1e-10,1]:
        print(tetra_vol(step_len + eps*step_len.grad))


# In[ ]:


def tetra_distortion(EndTip):
    cur_V = torch.cat([thV, EndTip])
    all_std = torch.einsum('ab,cbd->cad', torch.from_numpy(np.hstack([-np.ones((3,1)), np.eye(3)])), cur_V[T])
    all_jacob = torch.bmm(TG,all_std)
    dets = torch.det(all_jacob)
    print(all_jacob.shape)
    frob = torch.sum(all_jacob**2).sum(2).sum(1)
    return frob / dets
    #all_vol = -torch.logdet(all_std).sum()
    #return all_vol
    energy = -torch.logdet(all_jacob).sum()
    #energy = torch.sum((all_jacob - torch.eye(3).double())**2) - 1*torch.logdet(all_jacob).sum()
    return energy


# In[ ]:


EndTip.grad.data.zero_()
tetra_distortion(EndTip).backward()


# In[ ]:


with torch.no_grad():
    print('E0', tetra_distortion(EndTip).item())
    for eps in [1e-8,1e-9,1e-10]:
        print(tetra_distortion(EndTip - eps*EndTip.grad))

