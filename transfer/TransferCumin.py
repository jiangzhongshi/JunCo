#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys

from vis_utils import *

sys.path.append('../python')

import seism

pc = seism.PrismCage('../tests/data/bichon/armadillo/arma_socks.obj.h5')

mV, mF,refV, cp = h5reader('../tests/data/bichon/armadillo/arma_socks.obj.h5',
         'mV', 'mF', 'ref.V', 'complete_cp')


# In[8]:


fid, fuv = pc.transfer(mV,mF,refV)


# In[9]:


V_msh, T_msh = h5reader('../tests/data/bichon/armadillo/arma_socks.obj.h_tetshell_stitch.h5optim.h5', 'lagr','cells')


# In[10]:


mv_dict = {tuple(cp[f,e]):mF[f][e] for f in range(len(mF)) for e in range(3)} # Important: use cp here.


# In[11]:


import scipy
import scipy.spatial


# In[12]:


mv_tree = scipy.spatial.cKDTree(cp[:,:3].reshape(-1,3))


# In[14]:


bnd_msh = np.unique(igl.boundary_facets(T_msh[:,:4]))


# In[15]:


queries = [mv_tree.query(V_msh[i]) for i in bnd_msh]


# In[16]:


vmsh2mv = -np.ones(len(V_msh))
vmsh2mv[bnd_msh] = [mF[i//3, i%3] for _,i in queries]


# In[17]:


mapped_Tmsh = vmsh2mv[T_msh[:,:4]]


# In[18]:


tet_map = {}
for i in range(len(mapped_Tmsh)):
    t = list(mapped_Tmsh[i])
    if -1 in t:
        t.remove(-1)
        if len(t) != 3 or -1 in t:
            continue
        tet_map[tuple(sorted(t))] = i
    else:
        for j in range(4):
            tet_map[tuple(sorted(np.roll(t,j)[:-1]))] = i


# In[19]:


face_te = []
for fi, f in enumerate(mF):
    ti = tet_map[tuple(sorted(f))]
    face_te.append((ti, [list(mapped_Tmsh[ti]).index(i) for i in f]))


# In[20]:


out_mat = []
for fi, uv in zip(fid, fuv):
    bc = np.array([1-uv.sum(), uv[0],uv[1]])
    ti, ep = face_te[fi]
    tbc = np.zeros(4)
    tbc[ep] = bc
    tbc[0] = ti # hacky overwrite the first component
    out_mat.append(tbc)
out_mat = np.array(out_mat)


# In[21]:


tid = out_mat[:,0].astype(np.int)
tuv = out_mat[:,1:]
pts1 = np.einsum('vji,vj->vi', V_msh[T_msh[tid,:4]],
                 np.hstack([1-tuv.sum(axis=1)[:, None], tuv]))


# In[22]:


pts = np.einsum('vji,vj->vi', cp[fid,:3],
                    np.hstack([1-fuv.sum(axis=1)[:, None], fuv]))


# In[23]:


np.linalg.norm(pts - pts1)


# with open('../tests/data/bichon/armadillo/tid_bc.txt', 'w') as fp:
#     fp.write('\n'.join([' '.join([str(j) for j in i ])  for i in out_mat]))

# In[43]:


inpV,refF= h5reader('../tests/data/bichon/armadillo/armadillo.obj.h5','ref.V','ref.F')


# In[24]:


disp = np.loadtxt('../tests/data/bichon/armadillo/disp.txt')


# In[36]:


disp = disp.reshape(-1,4,3)


# In[26]:


import itertools


# In[37]:


allperms = []
for i,t in enumerate(T_msh):
    perm_list = [(np.linalg.norm(V_msh[t[:4]][list(perm)] - disp[i]), perm) for perm in (list(itertools.permutations(range(4))))]
    mpl = min(perm_list)
    assert mpl[0]< 1e-5
    allperms.append(mpl[1])


# In[38]:


perm_out = []
for t,u,v,w in out_mat:
    bc = np.array([1-u-v-w, u,v,w])
    bc = bc[list(allperms[int(t)])]
    bc[0] = t
    perm_out.append(bc)


# In[39]:


with open('../tests/data/bichon/armadillo/tid_bc_perm.txt', 'w') as fp:
    fp.write('\n'.join([' '.join([str(j) for j in i ])  for i in perm_out]))


# In[41]:


get_ipython().system('realpath ../tests/data/bichon/armadillo/tid_bc_perm.txt')


# In[157]:


out_mat


# In[52]:


ref_disp3 = np.loadtxt('../tests/data/bichon/armadillo/disp-p3.txt')


# In[68]:


ref_disp1 = np.loadtxt('../tests/data/bichon/armadillo/disp.txt')


# In[69]:


np.linalg.norm(ref_disp1 - ref_disp3,axis=1).max()


# In[71]:


igl.write_triangle_mesh('../tests/data/bichon/arma-socks/transfered.obj', pc.refV + ref_disp1, pc.refF)


# In[ ]:




