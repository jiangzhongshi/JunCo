#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import igl
from vis_utils import *
import numpy as np
import glob


# In[ ]:


import json
with open('/home/zhongshi/data/lind/resultsloop.json') as fp:
    our_results = json.load(fp)


# In[ ]:


with open('/home/zhongshi/data/lind/resultstree.json') as fp:
    mm_results = json.load(fp)


# # Times

# In[ ]:


alltimes= []
num_face = []
for i, j in our_results.items():
    if 'succeed' not in j or not j['succeed']:
        continue
    alltimes.append(j['duration']+1)
    num_face.append(j['vnum_ini'])


# In[ ]:


histocolor = 'rgb(162, 155, 254)'
def loghistogram(data):
    mi,ma = np.floor(np.log10(data).min()), np.ceil(np.log10(data).max())
    ticks = np.arange(mi,ma+1)
    fig = go.Figure(go.Histogram(x=np.log10(data),histnorm='percent',marker=dict(color=histocolor)),
                   layout=dict(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(gridcolor='grey',gridwidth=0.5,zeroline=True),
                      geo=dict(showframe=True)))
    fig.update_xaxes(
    ticktext=[f'1e{int(i)}' for i in ticks],
    tickvals=ticks)
    return fig


# In[ ]:


go.Figure(go.Box(y=alltimes)).update_layout(yaxis_type="log")


# In[ ]:


go.Figure(go.Scatter(x=num_face,y=(alltimes),mode='markers',
                     marker_line_width=2, marker_size=8,
                     marker=dict(color=histocolor))).update_layout(xaxis_type="log", yaxis_type="log")


# In[ ]:


np.save('alltimes.npy', alltimes)


# In[ ]:


get_ipython().system('realpath alltimes.npy')


# In[ ]:


mi,ma = np.floor(np.log10(alltimes).min()), np.ceil(np.log10(alltimes).max())
ticks = np.arange(mi,ma+1)
fig = go.Figure(go.Histogram(x=np.log10(alltimes),
                             histnorm='percent',
                             nbinsx=20,
                             marker=dict(color='rgb(150,160,246)')))
fig.update_layout(yaxis_type="log",
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  yaxis=dict(gridcolor='grey',gridwidth=0.5,zeroline=True),
                  geo=dict(showframe=True))
fig.update_layout(bargap=0.1)
#fig.update_layout(barmode='overlay')
# fig.update_traces(opacity=0.6)
fig.update_xaxes(
    ticktext=[f'1e{int(i)}' for i in ticks],
    tickvals=ticks)


# In[223]:


import os.path
from os import path
fold_list = (glob.glob('/home/lind/top_copy/**/CC*/'))


# In[207]:


allfiles = """00000120  00000583  00001248  00001838  00003002  00003700  00004562  00005245  00006127  00006775  00007595  00008104  00008848  00009405  00009794
00000248  00000606  00001332  00002178  00003107  00003787  00004865  00005676  00006239  00007028  00007736  00008150  00008970  00009406  00009828
00000334  00000928  00001372  00002325  00003123  00003875  00005003  00005877  00006259  00007088  00007824  00008171  00009046  00009490  00009837
00000343  00001137  00001470  00002394  00003248  00003965  00005145  00005996  00006260  00007132  00007836  00008404  00009057  00009506  00009856
00000345  00001162  00001654  00002614  00003382  00004112  00005150  00006032  00006313  00007252  00007877  00008570  00009234  00009507  00009902
00000395  00001232  00001655  00002705  00003502  00004314  00005157  00006061  00006546  00007359  00007943  00008739  00009235  00009692  00009956
00000489  00001233  00001825  00002786  00003535  00004321  00005158  00006076  00006724  00007552  00008091  00008796  00009398  00009767""".split()


# In[208]:


running = """ 00000334
 00000343
 00000395
 00001137
 00001233
 00001248
 00001332
 00001372
 00001470
 00001654
 00001655
 00002614
 00003535
 00003700
 00004112
 00004314
 00004321
 00004865
 00006032
 00006061
 00006775
 00007028
 00007088
 00008150
 00008570
 00008796
 00008848
 00009046
 00009398
 00009405
 00009406
 00009490
 00009506
 00009507""".split()


# In[213]:


finished =  list(set(allfiles) - set(running))


# In[214]:


finished[0]


# In[217]:


folder_names = []
for f in finished:
    folder_names += glob.glob('/home/lind/top_copy/'+f+'/CC*/')


# In[224]:


import tqdm


# In[228]:





# In[225]:


all_data = []
for f in tqdm.tqdm(folder_names):
    if not path.exists(f+'/resultloop.json'):
        continue
    label = igl.read_dmat(f+'FL_loop.dmat')
    missed = np.where(label==-1)[0]
    V,F = igl.read_triangle_mesh(f+'debug_mesh.obj')
    M = igl.doublearea(V,F)
    all_data.append((f, M.sum()/2, M[missed].sum()/2))


# In[236]:


get_ipython().system('cat /home/lind/top_copy/00002325/CC0/resultloop.json')


# In[239]:


import os
import os.path


# In[248]:


f


# In[249]:


get_ipython().system('cat /home/lind/top_copy/00005003/CC21/resultloop.json')


# In[247]:


"consistant topology\":true


# In[257]:


cleaner_data = []
for f,i,j in all_data:
    with open(f+'resultloop.json') as fp:
        lines = [l for l in fp.readlines()]
        if len(lines)==0:
            continue
    if r'{"consistant topology":false}' in lines[0]:
        cleaner_data.append((f,i,j))


# In[262]:


np.array([j/i for _,i,j in cleaner_data])


# In[265]:


len(cleaner_data)


# In[ ]:


fold_list = ['/home/lind/fail_copy/'+'/CC'.join(n.split('CC'))+'/' for n in names.split()]


# In[ ]:


missed_ratio = [j/i for _,i,j in all_data]


# In[ ]:


np.save('missed_area.npy', missed_ratio)


# In[152]:


import matplotlib.cm as cm


# In[161]:


import pymesh


# In[ ]:




