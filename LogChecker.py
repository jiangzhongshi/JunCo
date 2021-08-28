#!/usr/bin/env python
# coding: utf-8

# In[2]:


import shutil
import glob
import os
import igl
import numpy as np
from datetime import datetime
import re
from plotly import graph_objects as go
import pandas


# In[3]:


Thingi_data = pandas.read_pickle('10k_data.pickle')
abc_data = pandas.read_pickle('abc_data.pickle')


# In[26]:


(Thingi_data['time'].sum() + abc_data['time'].sum())/(len(Thingi_data) + len(abc_data))


# In[28]:


abc_data['time'].max()/3600


# In[8]:


with open('/home/zhongshi/data/abccomplete.log') as fp:
    complete = [l.rstrip() for l in fp.readlines()]


# In[61]:


import tqdm
vnum = 0
snum =0
for f in tqdm.tqdm(complete):
    with open('/home/zhongshi/data/0509abclog/'+f) as fp:
        lines = ' '.join(fp.readlines()[:20])
        initial_vnum = int(re.search(r'V=(.*),',lines).group(1))
        singular = re.search(r'Singularity ([0-9]*)',lines)#.group(1)
        if singular is None:
            singular = 0
        else:
            singular = int(singular.group(1))
        vnum += initial_vnum
        snum += singular


# In[114]:


len(ends.split('\n')),len(starts.split('\n'))


# ## timers

# In[9]:


def read_log(fp):
    lines = fp.readlines()
    if 'Memory' not in lines[-1]:
        return None
    start = None
    for i, l in enumerate(lines[::-1]):
         if re.search('Pass Precond', l) is not None:
            start = datetime.strptime(l[:25], '[%Y-%m-%d %H:%M:%S.%f]')
            break
    end = datetime.strptime(lines[-2][:25], '[%Y-%m-%d %H:%M:%S.%f]')
    initial_fnum = int(re.search(r'F=(.*)',' '.join(lines[:10])).group(1))
    bevel_fnum = int(re.search(r'-> [0-9]*/(.*)',' '.join(lines[:25])).group(1))
    for i in lines[::-1]:
        if 'Total Q' in i:
            last_info = i
            break
    info = re.search(r'number (.*), divide (.*), max (.*)\n',last_info)
    if info is None:
        info = re.search(r'fnum (.*), avg (.*), max (.*)\n',last_info)

    split_info = last_info.split()
    endF= int(info.group(1))
    meanQ= float(info.group(2))
    maxQ= float(info.group(3))
    time=((end-start).total_seconds())
    memory = int(lines[-1].split()[-1])
    return dict(fnum=initial_fnum,endf=endF,meanQ=meanQ,maxQ=maxQ,time=time,memory=memory,bevelF=bevel_fnum)


# In[10]:


import tqdm
collected_data = pandas.DataFrame(index=complete, columns=['fnum', 'endf', 'meanQ', 'maxQ', 'time', 'memory', 'bevelF'])
for f in tqdm.tqdm(complete):
    with open('/home/zhongshi/data/0519abclog/'+f) as fp:
        collected_data.loc[f] = read_log(fp)


# In[24]:


collected_data.to_pickle('abc_data.pickle')


# In[11]:


collected_data = pandas.read_pickle('10k_data.pickle')


# In[2]:


collected_data = pandas.read_pickle('abc_data.pickle')


# In[87]:


len(collected_data)


# In[14]:


pandas.to_numeric(collected_data['meanQ']).idxmax()


# In[13]:


histocolor = 'rgb(162, 155, 254)'

go.Figure(go.Scatter(x=np.array(collected_data['fnum'].tolist()),y=(np.array(collected_data['endf'].tolist())),mode='markers',
                     marker_line_width=1.5, marker_size=8,
                     marker=dict(color=histocolor))).update_layout(xaxis_type="log",yaxis_type='log',
                                                                               paper_bgcolor='rgba(0,0,0,0)',
                                                                               plot_bgcolor='rgba(0,0,0,0)',
                                                                               yaxis=dict(gridcolor='grey',gridwidth=0.5,zeroline=True),
                                                                               geo=dict(showframe=True)).write_image('abc_after_fnum.svg')


# In[14]:


go.Figure(go.Scattergl(x=collected_data['fnum'].to_numpy(),
                     y=collected_data['time'].to_numpy(),
                     mode='markers',
                     marker_line_width=1.5, marker_size=8,
                     marker=dict(color=histocolor,opacity=0.9))).update_layout(xaxis_type="log",yaxis_type='log',
                                                                               paper_bgcolor='rgba(0,0,0,0)',
                                                                               plot_bgcolor='rgba(0,0,0,0)',
                                                                               yaxis=dict(gridcolor='grey',gridwidth=0.5,zeroline=True),
                                                                               geo=dict(showframe=True)).write_image('abc_time_fnum.svg')


# In[ ]:


import plotly.express as px


# In[ ]:


import plotly
plotly.io.orca.config.use_xvfb = True


# In[15]:


go.Figure(go.Scatter(x=collected_data['fnum'],y=collected_data['memory']/1024**2,mode='markers',
                    marker_line_width=1.5, marker_size=8,
                     marker=dict(color=histocolor))).update_layout(xaxis_type="log",yaxis_type='log',
                                                                               paper_bgcolor='rgba(0,0,0,0)',
                                                                               plot_bgcolor='rgba(0,0,0,0)',
                                                                               yaxis=dict(gridcolor='grey',gridwidth=0.5,zeroline=True),
                                                                               geo=dict(showframe=True)).write_image('abc_mems.svg')


# In[ ]:




