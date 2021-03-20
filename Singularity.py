#!/usr/bin/env python
# coding: utf-8

# In[1]:


import igl
from vis_utils import *
import pandas
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


igl.read_triangle_mesh('~/Works')


# In[2]:


mV,mF,fT,fV,fF,birth = h5reader('../buildr/complex_original_temp.h5', 'mV', 'mF', 'fT', 'fV','fF', 'birth')


# In[3]:


mplot(mV,mF)


# In[2]:


mV,mF,fV,fF,fT, birth=h5reader('../buildr/trim_top_shell.h5',['mV', 'mF', 'fV', 'fF','fT', 'birth'])


# In[120]:


igl.write_triangle_mesh('temp.obj',fV,fF[mF.shape[0]:] )


# In[113]:


mplot(fV,fF[mF.shape[0]:])


# In[115]:


mV.shape[0], fV.shape[0], mF.shape[0], fF.shape[0]


# In[119]:


plt = mplot(mV,mF,wireframe=True,width=1000,height=1000)
#plt.add_lines(fV, mV[birth])


# In[20]:





# In[75]:


df = pandas.DataFrame(data={'vol':volume(fV,fT)})


# In[76]:


df['vol'].idxmin()


# In[78]:


fT[1098]


# In[22]:





# In[77]:


fig = px.box(df,y='vol',points="all")
fig.show()


# In[ ]:




