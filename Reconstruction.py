#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vishelper import *


# In[2]:


import pymesh


# In[3]:


mesh = pymesh.load_mesh('/home/zhongshi/Eagle_final.ply')


# In[4]:


pymesh.save_mesh('../buildr/eagle.obj',mesh)


# In[4]:


def put_in_unit_box(V):
    center = (V.max(axis=0) + V.min(axis=0))/2
    V -= center
    V/=V.max()
    return V


# In[5]:


V,F = put_in_unit_box(mesh.vertices.copy()), mesh.faces


# In[6]:


pV,pF = igl.upsample(V,F,4)


# In[7]:


with h5py.File('eagle_transfer.h5','w') as fp:
    fp['mV'] = V
    fp['mF'] = F
    fp['mC'] = np.vstack([mesh.get_attribute('vertex_red'), mesh.get_attribute('vertex_green'),mesh.get_attribute('vertex_blue')]).T
    fp['secV'] = V
    fp['secF'] = F
    fp['pV'] = pV
    fp['pF'] = pF


# In[8]:


refV,refF,sB,sT,sV,sF, _ =h5deser('../buildr/Eagle_final.ply.h5')


# In[16]:


sB[:9,:]


# In[18]:


plt=mplot(sB,sF)
plt.add_edges(sT, f2e(sF))
plt.add_points(np.array([[-0.00055636548298597995, 0.30200250133585743, -0.10621252687846758]]), shading=dict(point_size=0.01))
plt.add_points(sV[:9], shading=dict(point_size=0.05))


# In[78]:


mV,mF,mC, secV, secF, pV, pF, fid, qUV = h5reader('eagle_transfer.h5', ['mV', 'mF', 'mC', 'secV', 'secF', 'pV', 'pF', 'fid','qUV'])


# In[79]:


qu, qv = qUV[:,:1],qUV[:,1:]


# In[99]:


np.savez('temp.npz', mV= mV@np.diag([-1,-1,1]), mF=mF, mC = mC/255,
        pV =pV@np.diag([-1,-1,1]), pF=pF,fid = fid, qu=qu,qv=qv)


# In[84]:


plt = None
sh = dict(wireframe=False,flat=True,width=800,height=800)

vw = mp.Viewer(sh)
vw.add_mesh(mV@np.diag([-1,-1,1]),mF,c=mC/255,shading=sh)
plt = mp.Subplot(plt, vw, s=[1,2,0])

sh = dict(wireframe=False,flat=True,width=800,height=800)
vw = mp.Viewer(sh)
weighted_colors = (mC[mF[fid, 0]]*(1-qu-qv) + qu* mC[mF[fid, 1]]+mC[mF[fid, 2]]*qv)
vw.add_mesh(pV@np.diag([-1,-1,1]),pF,c=weighted_colors/255,shading=sh)
#vw.add_edges(secV,f2e(secF))
plt = mp.Subplot(plt, vw, s=[1,2,1])


# In[90]:





# In[118]:


plt = mplot(secV,secF,wireframe=False)
plt._Viewer__objects[0]['mesh'].geometry.attributes['normal'] = np.ones((secV.shape[0],3))
plt


# In[ ]:




