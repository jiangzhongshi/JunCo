#!/usr/bin/env python
# coding: utf-8

# In[2]:


from vis_utils import *

import glob
import pandas as pd
import tqdm
import plotly.express as px

import scipy.stats


# In[ ]:


files = glob.glob('/home/zhongshi/data/0103_cumin/*.obj.h5')

records = [dict(name=f) for f in files]
col_files = []
for i,f in enumerate(tqdm.tqdm(files)):
    fc = sorted([f for f in glob.glob(f+'_col?.h5')])
    if len(fc) == 0:
        continue
    col_files.append(fc[-1])


# In[22]:


with open('/home/zhongshi/public/shell_4layer/1222_extrude/files.list','w') as fp:
    fp.write('\n'.join(col_files))


# In[3]:


files = glob.glob('/home/zhongshi/data/0103_cumin/*.obj.h5')

records = [dict(name=f) for f in files]
for i,f in enumerate(tqdm.tqdm(files)):
    
    refF, mV,mF = h5reader(f,'ref.F','mV','mF')
    qualities = np.array([triangle_quality(*mV[f])for f in mF])
    records[i].update(dict(qmax=qualities.max(), qmean =qualities.mean(),
                          qmedian = np.median(qualities), Fsize=len(mF), inFsize =len(refF)))
#     fc = sorted([f for f in glob.glob(f+'_col?.h5')])[-1]
#     mV,mF = h5reader(fc,'mV','mF')
#     qualities = np.array([triangle_quality(*mV[f])for f in mF])
    records[i].update(dict(qcmax=qualities.max(), qcmean =qualities.mean(),
                          qcmedian = np.median(qualities)))
    break
#df = pd.DataFrame(records)


# In[4]:


qualities


# In[27]:


px.scatter(df,x='inFsize', y='qmax',hover_data=['name'],log_y=True, log_x=True)


# In[28]:


df[df['name'].str.contains('anc')]


# In[23]:


with open('/home/zhongshi/data/0113_areas.txt','r') as fp:
    lines = [l.rstrip() for l in fp.readlines()]


# In[44]:


sane_models =np.array([float(l.split('Area')[1])>1e-8 for l in lines])


# In[45]:


np.count_nonzero(sane_models)


# In[47]:


sane_qs = [q for i,q in enumerate(gathered_qs) if sane_models[i]]


# In[49]:


np.count_nonzero([i<100 for i,n in sane_qs]),np.count_nonzero([i<50 for i,n in sane_qs])


# In[52]:


with open('/home/zhongshi/data/0112_qualities.txt','r') as fp:
    lines = [l.rstrip() for l in fp.readlines()]


# In[53]:


def process_line(l):
    name=  l.split('.log')[0]
    Q = float(l.split('max')[1])
    return Q,name
#sorted([process_line(l) for l in _36.split('\n')])


# In[54]:


gathered_qs = [process_line(l) for l in lines]


# In[55]:


sorted(gathered_qs)[::-1]


# In[10]:


import glob
import os


# In[13]:


set([os.path.basename(f) for f in glob.glob('/home/zhongshi/data/0113_Th_cumin/*.h5')]) - set([os.path.basename(f) for f in glob.glob('/home/zhongshi/data/0502_raw10k/*.h5')])


# In[ ]:




