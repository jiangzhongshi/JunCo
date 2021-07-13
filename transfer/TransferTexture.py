#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from vis_utils import *

sys.path.append('../python')

import seism
pc = seism.PrismCage('../tests/data/bichon/microstructure.obj.h5')


# In[9]:


mV, mF,refV, cp,refF,inpV = h5reader('../tests/data/bichon/microstructure.obj.h5',
         'mV', 'mF', 'ref.V', 'complete_cp','ref.F','inpV')


# In[3]:


from curve import fem_tabulator


# In[37]:


import imageio


# In[39]:


texture = imageio.imread('/home/zhongshi/spot_texture.png')


# In[21]:


V,TC,VN, F,FTC, FTN= igl.read_obj('/home/zhongshi/spot_dense.obj')


# In[122]:


colors = []
for f,q in zip(tqdm.tqdm(fid), quv):
    bc = np.array([1-q.sum(),q[0],q[1]])
    tcx, tcy = (TC[FTC[f]]).T
    tc = np.array([((1-tcy)*1024).astype(int), (tcx*1024).astype(int)]).T
    local_color = np.asarray(texture[tc[:,0],tc[:,1]])/255
    colors.append(np.clip(bc@local_color,0,1))


colors = np.array(colors)


# In[4]:


tri_o3 = fem_tabulator.tuple_gen(order=3, var_n=2)


# In[5]:


lin_cp = np.einsum('fed,Ee->fEd',cp[:,:3,:], np.array(tri_o3)/3)


# In[49]:


lV, lF, lE = highorder_sv(lin_cp, level=4)


# In[50]:


fid, quv = pc.transfer(refV,refF, lV.reshape(-1,3))


# In[51]:


eg = igl.exact_geodesic(inpV,refF.astype(np.long), vs = np.array([[0]]).astype(np.long), vt = np.arange(len(inpV)).astype(np.long))


# In[52]:


trans_eg = np.einsum('vji,vj->vi',eg[refF[fid]][:,:,None], np.hstack([1-quv.sum(axis=1)[:, None], quv]))


# In[116]:


import tqdm


# In[53]:


hV,hF,hE =highorder_sv(cp, level=4)


# In[44]:


from matplotlib import cm


# In[54]:


colors=  cm.terrain_r(trans_eg.flatten()/trans_eg.max())[:,:3]


# In[55]:


colors


# In[56]:


p= mp.plot(hV.reshape(-1,3), hF,c=colors)
p.add_edges(hV.reshape(-1,3), hE)


# In[61]:


get_ipython().system('realpath ../tests/data/bichon/microstructure/micro_color.ply')


# In[60]:


write_obj_lines('../tests/data/bichon/microstructure/micro_color_edge.obj', hV.reshape(-1,3),hE)


# In[58]:


ply_writer('../tests/data/bichon/microstructure/micro_color.ply', hV.reshape(-1,3),hF, (colors*255).astype(np.int))


# In[66]:


with open('/home/zhongshi/spot_curve.obj','w+') as fp:
    fp.writelines(['v ' + ' '.join(map(str,list(v)))+'\n' for v in hV.reshape(-1,3)])
    fp.writelines(['vt ' + ' '.join(map(str,list(v)))+'\n' for v in uvcoord])
    fp.writelines([f'f {f[0]}/{tf[0]} {f[1]}/{tf[1]}  {f[2]}/{tf[2]}\n' for f,tf in zip(hF+1,hF+1)])


# In[ ]:


with open('./spot_curve_uv.obj','w') as fp:
    fp.write('')


# In[ ]:




