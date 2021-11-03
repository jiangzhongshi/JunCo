#!/usr/bin/env python
# coding: utf-8



from vis_utils import *




import igl




import numpy as np
import meshplot as mp




v,f = igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/bunny.off')




def cc_quad(v,f):
    verts = list(v)
    tt, tti = igl.triangle_triangle_adjacency(f)
    edge_id = - np.ones_like(tt)
    avail_id = len(v)
    for fi, _ in enumerate(tt):
        for ei in range(3):
            if edge_id[fi,ei] == -1:
                edge_id[fi,ei] = avail_id
                edge_id[tt[fi,ei],tti[fi,ei]] = avail_id
                v0, v1 = f[fi,ei], f[fi,(ei+1)%3]
                verts.append((v[v0]+v[v1])/2)
                avail_id += 1

    bc = igl.barycenter(v,f)
    quads = []
    for fi, [v0,v1,v2] in enumerate(f):
        pid = fi + avail_id
        for ei in range(3):
            v0, e0, e2 = f[fi,ei], edge_id[fi,ei], edge_id[fi,(ei+2)%3]
            quads += [[v0,e0,pid,e2]]
    return np.asarray(verts + list(bc)), np.asarray(quads)




def q2e(f):
    return np.array([f[:,i] for i in [[0, 1],[1,2],[2,3],[3,0]]]).reshape(-1,2)




vq,fq = cc_quad(v,f)



def greedy_pairing(f):
    tt, tti = igl.triangle_triangle_adjacency(f)
    occupied = -np.ones(len(f))
    qid = 0
    pairs = []

    for fi in range(len(f)):
        if occupied[fi] >= 0:
            continue
        for e in range(3):
            fo = tt[fi,e]
            if occupied[fo] >= 0:
                continue
            occupied[fi] = fo
            occupied[fo] = fi
            q = list(f[fi])
            q.insert(e+1, f[fo][tti[fi,e]-1])
            pairs.append(q)
            qid += 1
            break
    return occupied, pairs




sibling, pairs = greedy_pairing(f)




from collections import defaultdict
def crawling(trimesh, sibling, pairs):

    hybrid = list(trimesh[sibling==-1]) + list(pairs)
    connect = defaultdict(lambda:[None,None])
    def set_conn(v0,v1,x):
        if v0<v1:
            connect[(v0,v1)][0] = x
        else:
            connect[(v1,v0)][1] = x
    def get_conn(v0,v1):
        return connect[(v0,v1)][0] if v0<v1 else connect[(v1,v0)][1]
    for fi, f in enumerate(hybrid):
        ne = len(f)
        for i in range(ne):
            set_conn(f[i], f[(i+1)%ne],fi)

    # BFS
    def bfs_find_tri(t:int):
        visited = set()
        visited.add(t)
        path = [(t,None)]
        bfs_queue = [path]
        while len(bfs_queue) > 0:
            p = bfs_queue[0]
            bfs_queue.pop(0)
            t = p[-1][0]
            ne = len(hybrid[t])
            for i in range(ne):
                e = hybrid[t][(i+1)%ne], hybrid[t][i]
                fo = get_conn(*e) # oppo
                if fo in visited:
                    continue
                bfs_queue.append(p + [(fo,e)])
                if len(hybrid[fo]) == 3:
                    return bfs_queue[-1]
        return None

    def merge_tri_quad(tri, edge, quad): # this is a backward path
        q, e0 = quad
        v0,v1 = edge
        qv = list(hybrid[q].copy())
#         print('In', tri, edge, qv)
        x = list(set(tri)-set(edge))[0] # opposite vertex
        qv.insert(qv.index(v0), x)
#         print('Pent',qv, e0)

        if len(qv) == 4:
            assert e0 is None
            return None, None, qv
        newtri = [qv[qv.index(e0[0])-1], e0[0], e0[1]]
        qv.remove(e0[0])
        # splitter edge
#         print('Out:', newtri, e0, qv)
        return newtri, e0, qv
    while True:
        bachelor = np.where(np.array([len(h) for h in hybrid]) == 3)[0]
        print(len(bachelor))
        if len(bachelor) == 0:
            return
        path = bfs_find_tri(bachelor[0])
        tid, edge = path[-1]
        tri = hybrid[tid]
        new_quads = []
        pid = len(path) - 2
        while tri is not None:
            tri, edge, new_q = merge_tri_quad(tri, edge, path[pid])
            pid -= 1
            new_quads.append(new_q)
#         print(0)
        ## Update Connectivity and Quads.
        for p,_ in path:
            f = hybrid[p]
            ne = len(f)
            for i in range(ne):
                set_conn(f[i], f[(i+1)%ne], None)
            hybrid[p] = []
        for f in new_quads:
            ne = len(f)
            for i in range(ne):
                set_conn(f[i], f[(i+1)%ne], len(hybrid))
            hybrid.append(f)
    return hybrid

cpath = crawling(f, sibling, pairs)






