#!/usr/bin/env python
# coding: utf-8

from curve import fem_tabulator as feta
from vis_utils import h5reader, highorder_sv
from quad import quad_curve_utils as qr
from quad import quad_utils
import meshplot as mp
import numpy as np
import igl

def eval_bc(verts, faces, bc_i, denom:int):
    vf = (verts[faces[bc_i[:,0]]])
    return np.einsum('sed,se->sd', vf, bc_i[:,1:])/denom
def simple_query(p):
    x, y = p[:,0], p[:,1]
    return np.vstack([x,y,x**2]).T

def bezier_fit_matrix(order : int, level : int) -> np.ndarray:
    std_x, std_y = qr.quad_tuple_gen(level).T
    bsv = qr.tp_sample_value(std_x/level, std_y/level, order=order)
    bsv = bsv.reshape(len(bsv),-1)
    return bsv

def main():
    V,F = igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/planexy.off')
    V = simple_query(V)
    def score(f,e):
        return - np.linalg.norm(V[F[f,e]] - V[F[f,(e+1)%3]]) # long edges are diffse first
    siblings, quads = quad_utils.greedy_pairing(F, score=score)
    t2q, q2t,trim_types, quads = qr.quad_trim_assign(siblings, F)

    level = 5
    order = 3
    bsv = bezier_fit_matrix(order, level)
    def query(t_bc_i, denom):
        pts = eval_bc(V, F, t_bc_i, denom=level)
        return simple_query(pts)

    quad_cp = qr.quad_fit(V,F,quads, q2t, trim_types, level, order, bsv, query)
    if siblings.min() >= 0:
        print('Perfect Match')
        return quad_cp, None
    
    new_v, known_cp, newquads = qr.solo_cc_split(V, F, siblings, t2q, quads, quad_cp, level, order)
    cc_cp = qr.constrained_cc_fit(V, F, siblings, newquads, known_cp, level, order, bsv, query)
    return quad_cp, cc_cp
