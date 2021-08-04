import numpy as np
import quad.quad_curve_utils as qr

def eval_bc(verts, faces, bc_i, denom:int):
    vf = (verts[faces[bc_i[:,0]]])
    return np.einsum('sed,se->sd', vf, bc_i[:,1:])/denom

def bezier_fit_matrix(order : int, level : int) -> np.ndarray:
    std_x, std_y = qr.quad_tuple_gen(level).T
    bsv = qr.tp_sample_value(std_x/level, std_y/level, order=order)
    bsv = bsv.reshape(len(bsv),-1)
    return bsv