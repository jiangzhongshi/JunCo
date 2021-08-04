"""Utilities for B-spline fitting for Quad Mesh, and its validity"""
import quad.quad_curve_utils as qr
import numpy as np
from spline import cubic_spline as csp
import scipy
import scipy.sparse
import collections
import tqdm
from curve import fem_tabulator as feta
import igl


def evaluator(x, table):
    """Generate a matrix A, where A*cp = values"""
    bu, iu = csp.BSplineSurface._bspev_and_c(x, table, csp.poly_coefs[0])
    A = np.zeros((len(x), len(table)+2))
    for k, (i, b) in enumerate(zip(iu, bu)):
        A[k, i] = b
    return A


def splitter(width):
    """Square full-rank evaluators, used for directly solving during subdivision."""
    dim = width+3
    table = np.asarray(csp.table_1d(width))
    A = evaluator(np.linspace(0, width, dim), table)
    A1 = evaluator(np.linspace(0, width/2, dim), table)
    A2 = evaluator(np.linspace(width/2, width, dim), table)
    denom = 1
    return A, A1, A2, denom
#     return (A*denom).astype(int),(A1*denom).astype(int),(A2*denom).astype(int), denom


def subd(coef):
    print(coef.shape)
    assert False, "TODO: concile width and coef"
    if subd.A is None:
        subd.A, subd.A1, subd.A2, subd.denom = splitter(width)
    c1 = np.linalg.solve(subd.A, subd.A1@coef)
    c2 = np.linalg.solve(subd.A, subd.A2@coef)
    return c1,c2
subd.A = None

def basis_row(x, table, du=0, dv=0):
    dim = len(table) + 2
    bu, iu = csp.BSplineSurface._bspev_and_c(x[:,0], table, csp.poly_coefs[0])
    bv, iv = csp.BSplineSurface._bspev_and_c(x[:,1], table, csp.poly_coefs[0])
    outer = np.einsum('bi,bo->bio', bu, bv).flatten()
    cols = (np.expand_dims(iu, 2) * dim + np.expand_dims(iv, 1)).flatten()
    rows = np.arange(iu.shape[0])[:, None].repeat(iu.shape[1]*iv.shape[1], axis=1).flatten()
    return rows, cols, outer
    
def bspline_fitting_matrix(width, level, num_reg=0.):
    """
    Generate the matrix for bspline fitting.
    replicating _global_basis_row from `cubic_spline.py`
    """
    X = np.array(qr.quad_tuple_gen(level))
    table = np.asarray(csp.table_1d(width))    
    dim = width+3
    rows, cols, outer = basis_row(X/level*width, table)
    A = scipy.sparse.csr_matrix((outer, (rows, cols)),
                                shape=(rows.max()+1, dim**2))
    reg_mat = None
    if num_reg > 0:
        def add_half(l):
            return [0.5] + list(range(l)) #+ [l-0.5]
#         num_reg = 2 # number of regularizer for each internal.
        regularizer = [[i,j] for i in add_half(num_reg*(width+1)) for j in add_half(num_reg*(width+1))]
        regularizer = np.array(regularizer)/num_reg
        r_rows, r_cols, r_outer, offset = [],[],[], 0

        for u in range(3):
            row0, col0, data0 = basis_row(regularizer, table, du = u, dv = 2-u)
            print(row0.dtype)
            r_rows = np.concatenate([r_rows, row0 + offset])
            r_cols = np.concatenate([r_cols, col0])
            r_outer= np.concatenate([r_outer, data0])
            offset += r_rows.max()+1

        reg_mat = scipy.sparse.csr_matrix((r_outer, (r_rows, r_cols)),
                                shape=(int(r_rows.max())+1, dim**2))

    return A, reg_mat

def constrained_ho_fit(quads, valid, q_cp, q2t,trim_types, known_cp, width:int, level:int, bsv, query):
    """
    Constrained High Order Elevated fitting, for invalid quads.
    """
    order = 2*width + 2

    assert level >= order
    newquads = quads[valid==False]
    F_sa = qr.quad_ho_F(newquads, level=level)
    F_or = qr.quad_ho_F(newquads, level=order)
    std_codec = qr.local_codecs_on_edge(order)
    edge_node_map = {tuple(sorted(k)): i
                     for i, k in enumerate(std_codec) if k[0] != -1}
    
    # break down input known_cp on edges (v0,v1) to Bernstein on level
    def converter(width):
        """Square full-rank evaluators, used for directly solving during subdivision."""
        dim = width+3
        table = np.asarray(csp.table_1d(width))
        table2 = np.asarray(csp.table_1d(width*2))
        A = evaluator(np.linspace(0,width,2*width+3),table)
        A1 = evaluator(np.linspace(0,2*width,2*width+3),table2)
        # print(A.shape, A1.shape)
        return A, A1
    A,A1 = converter(width)
    
    known_dict = dict()
    new_cp = np.zeros((len(newquads), (order+1)**2,3))
    for q, quad in enumerate(tqdm.tqdm(newquads)):
        for qv0 in range(4):
            qv1 = (qv0+1)%4
            if (quad[qv0], quad[qv1]) not in known_cp:
                continue
#             print(f'Qv {(qv0,qv1)} QADU {(quad[qv0], quad[qv1])}')
            sub_cp = np.linalg.solve(A1, A@known_cp[(quad[qv0], quad[qv1])])
#             debug.save.append(sub_cp)
            tup_list = [tuple(sorted([qv0]*(order-t1) + [qv1]*(t1))) for t1 in range(order+1)]
            assigner = np.array([edge_node_map[t] for t in tup_list])
#             print(assigner)
            for i, t in enumerate(tup_list):
                known_dict[F_or[q,edge_node_map[t]]] = sub_cp[i]
    
    known_ids = np.ones(len(known_dict), dtype=int)
    known_vals = np.ones((len(known_dict), 3))
    for i, (k, v) in enumerate(sorted(known_dict.items())):
        known_ids[i], known_vals[i] = k, v
    # print(known_dict)    
    all_samples = np.zeros((F_sa.max() + 1, 3))
    rows, cols, vals = [], [], []
    bsv = bsv.tocoo()
    br, bc, bv = bsv.row, bsv.col, bsv.data

    q2t = q2t[valid==False]
    for q, (t0, t1) in enumerate(tqdm.tqdm(q2t, desc='Constrained Fit for Invalid Quads')):
        tbc0 = np.array(qr.sample_for_quad_trim(trim_types[t0], trim_types[t1], level),
                        dtype=int)
        tbc0[:, 0] = np.asarray([t0, t1])[tbc0[:, 0]]
        sample_vals = query(tbc0, denom=level)

        all_samples[F_sa[q]] = sample_vals
        sample_q, order_q = F_sa[q], F_or[q]
        rows = np.concatenate([rows, sample_q[br]])
        cols = np.concatenate([cols, order_q[bc]])
        vals = np.concatenate([vals, bv])
    ijk, idx = np.unique(np.vstack([rows, cols]).astype(int), axis=1, return_index=True)
    A = scipy.sparse.csr_matrix(arg1=(np.array(vals)[idx],
                                      ijk))
    assert A.shape[0] == all_samples.shape[0]
    sol = qr.quadratic_minimize(A, all_samples, (known_ids, known_vals))
#     print(known_ids)
    print('KnownRes:',np.linalg.norm(sol[known_ids] - known_vals))
#     print('0diff', np.linalg.norm(sol[[i*11 for i in range(11)]] - debug.save[0]))
    all_cp = sol[F_or]
    return all_cp, all_samples[F_sa], A@sol




def bsp_sv(cp, level, width, debug=False):
    def local_upsample(level: int):
        usV, usF = igl.upsample(np.array([[0, 0], [1, 0], [1, 1], [0, 1.]]),
                                np.array([[0, 1, 2], [0, 2, 3]]), level)
        bnd0 = igl.boundary_loop(usF)
        usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE

    assert len(cp.shape) == 3
    usV, usF, usE = local_upsample(level=level)
    trV = usV*width
    table = np.asarray(csp.table_1d(width))

    bu, iu = csp.BSplineSurface._bspev_and_c(trV[:,0], table, csp.poly_coefs[0])
    bv, iv = csp.BSplineSurface._bspev_and_c(trV[:,1], table, csp.poly_coefs[0])
    cols = (np.expand_dims(iu, 2) * (width+3) + np.expand_dims(iv, 1)).flatten()
    if debug:
        print(cp[:,cols,:].shape,width)
    sv = np.einsum('bj,fbjkd,bk->fbd',
                              bu, cp[:,cols,:].reshape(len(cp),len(bu),4,4,3), bv)
    return (sv,
            np.vstack([usF+i*len(usV) for i in range(len(cp))]),
            np.vstack([usE+i*len(usV) for i in range(len(cp))]))



def check_validity(mB,mT,F, quads, q2t, trim_types, quad_cp, width):
    """Check inversions

    Args:
        mB ([type]): [description]
        mT ([type]): [description]
        F ([type]): [description]
        quads ([type]): [description]
        q2t ([type]): [description]
        trim_types ([type]): [description]
        quad_cp ([type]): [description]
        width ([type]): [description]

    Returns:
        [type]: [description]
    """
    all_b, all_t = qr.top_and_bottom_sample(mB,mT, F, quads, q2t, trim_types, level=width)
    v4, f4 = qr.split_square(width)
    tup = feta.tuple_gen(order=4, var_n=2) # use cubic + 1 for tetrahedra
    grids = np.einsum('fed,Ee->fEd', v4[f4], np.asarray(tup))
    grid_ids = np.ravel_multi_index(grids.reshape(-1,2).T, dims = (4*width+1,4*width+1)).reshape(len(f4), -1)
    valid_quad = np.ones(len(quad_cp), dtype=bool)
    A13, _ = bspline_fitting_matrix(width, width*4)
    import tqdm
    for q,qcp in enumerate(tqdm.tqdm(quad_cp)):
        lagr = A13@qcp
        for t,g in zip(f4, grid_ids):
            if not (prism.elevated_positive_check(all_b[q][t], all_t[q][t], lagr[g], True)):
                valid_quad[q] = False
                break
    return valid_quad


