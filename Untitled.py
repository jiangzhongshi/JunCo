#!/usr/bin/env python
# coding: utf-8

# In[15]:


from vis_utils import *
import sys
sys.path.append('../python/')
from curve import fem_generator
from curve import surface_to_curved_layer


# In[3]:


with open('/home/zhongshi/fullname_10k_good.txt') as fp:
    lines = [l.rstrip() for l in fp.readlines()]


# In[4]:


import tqdm


# In[8]:


watertights = []
for f in tqdm.tqdm(lines):
    V,F = igl.read_triangle_mesh(f)
    if len(V) == 0 or len(F) == 0:
        continue
    SV, SVI,SVJ,_ = igl.remove_duplicate_vertices(V,F,0)
    V,F = SV, SVJ[F]
    TT,_ = igl.triangle_triangle_adjacency(F)
    if TT.min() >= 0: #bnd
        watertights.append(f)


# In[12]:


with open('/home/zhongshi/fullname_10k_closed.txt','w') as fp:
    fp.write('\n'.join([l.replace('/home/zhongshi/data','/scratch/work/panozzo/') for l in watertights]))


# In[9]:


len(watertights)


# In[133]:


V,F = igl.read_triangle_mesh('/home/zhongshi/data//gao_hex/cad/fandisk_input_tri.obj')


# In[136]:


FN = igl.per_face_normals(V,F,np.ones(3))
TT,TTi = igl.triangle_triangle_adjacency(F)
di_angles = (FN[TT]*FN[:,None,:]).sum(axis=2)
E =np.array([(F[f,e], F[f,e-2]) for f,e in zip(*np.where(di_angles < 0.5))])


# In[147]:


p = mp.plot(V,F,wireframe=True)
p.add_edges(V,E,line_color='red')


# In[22]:


import meshio


# In[23]:


m = meshio.read('/home/teseo/codt/skin-theirs.hom.msh')
mV, mT = m.points, m.cells[0].data


# In[ ]:





# In[57]:


def highorder_sv(cp,level=3, order=3):
    dim = 3
    def local_upsample(level:int):
        tet = igl.boundary_facets(np.array([0, 1, 2, 3]).reshape(1,4))
        vv = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
#         usV, usF = igl.upsample(np.eye(3)[:,1:], np.arange(3)[None,:], level)
        usV, usF = igl.upsample(vv, tet, level)
        bnd0 = igl.boundary_loop(usF)
        usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE
    usV,usF,usE = local_upsample(level=level)
    info = fem_generator.basis_info(order, dim)
    l2b = info['l2b']
    bas_val = fem_generator.bernstein_evaluator(usV[:,0],usV[:,1],usV[:,2], info['codec']).T
    print(cp.shape)
    sv = (bas_val@cp)
    return sv, np.vstack([usF+i*len(usV) for i in range(len(sv))]), np.vstack([usE+i*len(usV) for i in range(len(sv))])


# In[58]:


hV, hF, _ = highorder_sv(mV[mT])


# In[ ]:


p = mp.plot(hV.reshape(-1,3),hF)
p.add_points(mV, point_size=0.1)


# In[20]:


meshio.write('temp.msh', meshio.Mesh(points=(info3['b2l']@mV[mT]).reshape(-1,3),
                                     cells=[('tetra20', np.arange(20*len(mT)).reshape(-1,20))]))


# In[8]:


N = 20
def scheme(_): return ''
scheme.points = np.asarray(
    list(filter(lambda x: sum(x) == N, itertools.product(range(N+1), repeat=4))))/N
scheme.weights = np.ones(len(scheme.points))/len(scheme.points)
scheme_pts = scheme.points.T
cod = fem_generator.codecs()


# In[189]:


import quadpy

sheme_pts = quadpy.t3.schemes['witherden_vincent_01']().points


# In[9]:


info3= fem_generator.basis_info(order=3,nsd=3, force_codec='tetra20')


# In[10]:


d_bas_val = fem_generator.bernstein_deri_evaluator(*scheme_pts[1:], info3['codec'])


# In[16]:


jacs = np.einsum('fNd,eNs->fsed', info3['l2b']@mV[mT], d_bas_val)


# In[17]:


np.linalg.det(jacs).shape


# In[19]:


np.linalg.det(jacs).min()


# In[18]:


np.where(np.linalg.det(jacs)==0)


# In[132]:


idx = np.array(_127).T


# In[136]:


scheme.points[idx[:,1],1:].shape


# In[141]:


mV[mT[idx[:,0],:4]].shape


# In[142]:


neg_pts = np.einsum('fEd,fE->fd',
    mV[mT[idx[:,0],:4]], scheme.points[idx[:,1]])


# In[148]:





# In[145]:


neg_pts.shape


# In[150]:


p = mp.plot(mV,t2e(mT[:1,:4]))
p.add_points(neg_pts[idx[:,0]==0], point_size=0.01)


# In[119]:


p = mp.plot(scheme.points[_115, :1],point_size=0.8)
p.add_edges(np.eye(4)[:,1:], np.array([[0,1],[1,2],
                                       [0,2],[0,3],
                                      [1,3],[2,3]]))


# In[54]:


meshio.write('temp.msh', meshio.Mesh(points=mV,
                                     cells=[('tetra20', mT[:1])]))


# In[46]:


import glob


# In[53]:


with open('/home/zhongshi/autotool_mesh_stage_2_final.hom') as fp:
    lines = [l.split() for l in fp.readlines()]


# In[54]:


import meshplot as mp
import numpy as np


# In[55]:


vnum = int(lines[4][0])
cur = 5
print(vnum)
V = np.array([list(map(float, l))[:-1] for l in lines[cur:vnum+cur]])
cur += vnum
enum = int(lines[cur][0])
print(f'enum{enum}')
cur += 1
E = np.array([list(map(int, l)) for l in lines[cur:enum+cur]])
cur += enum
fnum = int(lines[cur][0])
cur += 1
F = np.array([list(map(int, l)) for l in lines[cur:fnum+cur]])
cur += fnum
tnum = int(lines[cur][0])
cur += 1
T = np.array([list(map(int, l)) for l in lines[cur:tnum+cur]])
cur += tnum


# In[56]:


hE = np.array([list(map(int,l)) for l in lines[cur:] if len(l)==4])
hF = np.array([list(map(int,l)) for l in lines[cur:] if len(l)==5])


# In[57]:


tetra20_codec = '000,111,222,333,001,011,112,122,022,002,033,003,233,223,133,113,012,013,023,123'
tetra20_codec_n = np.array([list(map(int, c))
                                for c in tetra20_codec.split(',')])


# In[58]:


base_cp = np.einsum('fed,Ee->fEd', V[T],
          np.eye(4)[tetra20_codec_n].mean(axis=1))
#.shape


# In[91]:


import scipy.spatial
tree = scipy.spatial.KDTree(np.vstack([V, np.einsum('fed, fe->fd',V[E[hE[:,1]]],hE[:,[3,2]])/3, np.einsum('fed, fe->fd',V[F[hF[:,1]]],hF[:,2:])/3]))


# In[92]:


ind_list = np.concatenate([np.arange(len(V)), hE[:,0].flatten(), hF[:,0].flatten()])
gmshT = np.zeros(len(T)*20, dtype=np.int)
alldist = []
for i,b in enumerate(base_cp.reshape(-1,3)):
    _, ind = (tree.query(b))
    gmshT[i] = ind_list[ind]
    alldist.append(_)
    ##for 


# In[93]:


np.max(alldist)


# In[94]:


import meshio
meshio.write('brave.msh',
meshio.Mesh(points=V, cells=[('tetra20',gmshT.reshape(-1,20))])
)


# In[100]:


newV = (bas33['b2l']@V[gmshT.reshape(-1,20)])

meshio.write('brave.msh',
meshio.Mesh(points=newV.reshape(-1,3), cells=[('tetra20',np.arange(20*len(T)).reshape(-1,20))])
)


# In[90]:


get_ipython().system('realpath brave.msh')


# In[67]:





# In[72]:


bas33 = fem_generator.basis_info(order=3,nsd=3,force_codec='tetra20')


# In[ ]:




