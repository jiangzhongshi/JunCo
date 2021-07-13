#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vis_utils import *
import glob
import os
import sys


# In[3]:


sys.path.append('../python')

from curve import fem_tabulator


# In[4]:


# Gather cods
tri_o4 = fem_tabulator.tuple_gen(order=4, var_n=2)
tet_o4 = fem_tabulator.tuple_gen(order=4, var_n=3)
def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i]*j]
tri_o4_n = np.array([codec_to_n(c) for c in tri_o4])
tet_o4_n = np.array([codec_to_n(c) for c in tet_o4])


# In[ ]:


assert False, "add L2B!!!!"


# In[5]:


def clip_and_color(V,T,A=0.5,B =0, C=0.5,D=-0.4):
    x, y, z = V[T].mean(axis=1).T # barycentric
    clip_index = np.where(A*x + B*y + C*z + D< 0)[0]

    fc = igl.boundary_facets(T[clip_index])

    original_faces = set(frozenset(f) for f in igl.boundary_facets(T))
    from_facet = [i for i, f in enumerate(
        fc) if frozenset(f) in original_faces]
    comple_ff = np.setdiff1d(np.arange(len(fc)),from_facet)
    return fc, from_facet, comple_ff


# In[132]:


V_msh, T_msh = h5reader('../tests/data/bichon/armadillo/arma_socks.obj.h_tetshell_stitch.h5', 'lagr','cells')


# In[92]:


igl.boundary_facets(T_msh[:,:4]).shape


# In[83]:


tup2node = {tuple(sorted(f[j])):T_msh[i,j] for i,f in enumerate(T_msh[:,tet_o4_n]) for j in range(35)}        


# # Linear Slice Test

# In[99]:


Fbnd, slic_f, comp_f = clip_and_color(V_msh, T_msh[:1298*3,:4], 0.6, 0, 0.5, -0.46)


# In[110]:


Flinear, _, _ = clip_and_color(V_msh, T_msh[1298*3:,:4], 0.6, 0, 0.5, -0.48)


# In[124]:


write_obj_lines('../tests/data/bichon/arma-socks/stitch_slice_C_edge.obj', V_msh, igl.edges(Flinear))


# In[111]:


p = mp.plot(V_msh, Fbnd[slic_f])
p.add_mesh(V_msh, Fbnd[comp_f], c=np.array([0.7,0.7,1]),wireframe=True)
p.add_mesh(V_msh, Flinear, c=np.array([0.5,0.5,0.5]),wireframe=True)


# In[63]:


@mp.interact(ev=0.5, step=0.01)
def sf(ev):
    Fbnd, slic_f, comp_f = clip_and_color(V_msh, T_msh[:,:4], 0.6, 0, ev,-0.46)
    p.update_object(oid =0,vertices=V_msh, faces=Fbnd[slic_f]);
    p.update_object(oid =1,vertices=V_msh, faces=Fbnd[comp_f]);


# # Elevate Order

# In[112]:


p4F0 = np.array([tup2node[tuple(sorted(f))] for f in Fbnd[slic_f][:,tri_o4_n].reshape(-1,4)]).reshape(-1,15)


# In[113]:


p4F1 = np.array([tup2node[tuple(sorted(f))] for f in Fbnd[comp_f][:,tri_o4_n].reshape(-1,4)]).reshape(-1,15)


# In[127]:


tri4_info = fem_tabulator.basis_info(order=4,nsd=2)


# In[128]:


tri4_info['l2b']@(V_msh[p4F0])


# In[125]:


V_msh[p4F0].shape


# In[129]:


hV,hF,hE = highorder_sv(tri4_info['l2b']@V_msh[p4F0], 
                        level=3, 
                        order=4)


# In[130]:


hV1,hF1,hE1 = highorder_sv(tri4_info['l2b']@V_msh[p4F1], level=3, order=4)


# In[131]:


p = mp.plot(hV.reshape(-1,3),hF,wireframe=False)
p.add_edges(hV.reshape(-1,3),hE)
p.add_mesh(hV1.reshape(-1,3),hF1,c=np.array([0.7,0.7,1]),wireframe=False)
p.add_edges(hV1.reshape(-1,3),hE1)
#p.add_mesh(V_msh, Flinear, c=np.array([0.5,0.5,0.5]),wireframe=True)


# # Export and Dump Region

# In[79]:


def write_curve(filename, cp, level=3):
    hV,hF,hE = highorder_sv(cp,level=level ,order=4)
    igl.write_triangle_mesh(filename + '_tri.obj', hV.reshape(-1,3), hF)
    write_obj_lines(filename + '_edge.obj', hV.reshape(-1,3), hE)


# In[133]:


write_curve('../tests/data/bichon/arma-socks/stitch_slice_A', tri4_info['l2b']@V_msh[p4F1],level=5)
write_curve('../tests/data/bichon/arma-socks/stitch_slice_B', tri4_info['l2b']@V_msh[p4F0], level=5)


# In[119]:


igl.write_triangle_mesh('../tests/data/bichon/arma-socks/stitch_slice_C.obj', V_msh, Flinear)


# In[47]:


p = mp.plot(hV.reshape(-1,3), hF,wireframe=False)
p.add_edges(hV.reshape(-1,3),hE)
p.add_mesh(hV1.reshape(-1,3), hF1,wireframe=False,c=np.array([0.7,0.7,1]))
p.add_edges(hV1.reshape(-1,3),hE1)


# In[58]:


p = mp.plot(V,fc[ff])
p.add_mesh(V,fc[comple_ff],c=np.array([0.7,0.7,1]))
p.add_points(
    np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
              [1.77,1.42,1.39]]), point_size=1.0,c=np.arange(5))


# In[ ]:




