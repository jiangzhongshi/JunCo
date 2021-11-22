#!/usr/bin/env python
# coding: utf-8

def mark_seams():
    V,TC,_,F,FTC,_ = igl.read_obj('/home/zhongshi/AirCraftMark.obj')

    TT,TTi = igl.triangle_triangle_adjacency(FTC)

    # add all boundary edges from FTC (texture seams)
    edges = []
    for i in range(len(TT)):
        for j in range(3):
            if TT[i,j] == -1:
                edges.append((F[i,j],F[i,j-2]))


    uE = np.unique(np.sort(np.array(edges),axis=1),axis=0)

    with h5py.File('/home/zhongshi/Aircraft/feat_mark.h5', 'w') as fp:
        fp['E'] = uE
    ## System call for bichon generator.

    mV,mF, meta_f, meta_i = h5reader('../buildr/AirCraftMark.obj.h5.init','ref.V','ref.F','meta_edges_flat', 'meta_edges_ind')

    metas = np.split(meta_f, meta_i[1:-1])