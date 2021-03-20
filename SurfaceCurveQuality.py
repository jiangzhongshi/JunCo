#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vis_utils import *


# In[49]:


mV,mF,cp = h5reader('../tests/data/bichon/armadillo/arma_socks.obj.h5','mV','mF','complete_cp')


# In[67]:


np.max([triangle_quality(*p) for p in mV[mF]])


# In[ ]:


deri_u, deri_v = h5reader('../python/curve/data/tri_o3_lv3.h5','deri_u', 'deri_v')


# In[77]:


def curved_amips(cp, deri_u, deri_v):
    DU, DV = np.einsum('fed,se->fsd', cp, deri_u), np.einsum('fed,se->fsd', cp, deri_v)

    e1_len = np.linalg.norm(DU,axis=2)
    e2_x = (DU*DV).sum(axis=2)/e1_len
    e2_y = np.linalg.norm(DV - DU * (e2_x/e1_len)[:,:,None],axis=2)

    inv_ref = np.array([[1,-1/np.sqrt(3)], [0, 2/np.sqrt(3)]])

    quality = ((e1_len**2 + (e2_x - e2_y/np.sqrt(3))**2 + (e2_y**2)*(4/3))/
               (e1_len * 2*e2_y/np.sqrt(3)))
    return quality


# In[88]:





# In[78]:


import glob
import os
import tqdm


# In[79]:


files = glob.glob('/home/zhongshi/data/0124_night/*.h5')


# In[92]:


hexfiles = glob.glob('/home/zhongshi/data/0124_hex_cumin/*.h5')


# In[113]:


records = [dict(name=os.path.basename(f)) for f in files+hexfiles]


# In[114]:


for i,f in enumerate(tqdm.tqdm(files+hexfiles)):
    with h5py.File(f,'r') as fp:
        cp = fp['complete_cp'][()]
        qualities = curved_amips(cp, du,dv)
        records[i].update(dict(qmax=qualities.max(), qmean =qualities.mean(),
                          qmedian = np.median(qualities), Fsize=len(cp)))


# In[97]:


import pandas as pd


# In[115]:


df = pd.DataFrame(records)


# In[117]:


df.to_pickle('/home/zhongshi/surface_qualities.pkl')


# In[62]:


DU, DV = np.einsum('fed,se->fsd', cp, deri_u), np.einsum('fed,se->fsd', cp, deri_v)


# In[63]:


e1_len = np.linalg.norm(DU,axis=2)
e2_x = (DU*DV).sum(axis=2)/e1_len
e2_y = np.linalg.norm(DV - DU * (e2_x/e1_len)[:,:,None],axis=2)

inv_ref = np.array([[1,-1/np.sqrt(3)], [0, 2/np.sqrt(3)]])

quality = ((e1_len**2 + (e2_x - e2_y/np.sqrt(3))**2 + (e2_y**2)*(4/3))/
           (e1_len * 2*e2_y/np.sqrt(3)))


# In[72]:


tri = np.array([[e1_len, e2_x],[0*e2_y, e2_y]])


# In[74]:


(tri@inv_ref).shape


# In[ ]:




