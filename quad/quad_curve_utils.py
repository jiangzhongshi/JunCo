import pdb
from curve import fem_generator as feta
import osqp
import numpy as np
from curve import fem_generator as fetaa
import igl
import scipy

ref_quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])


def quadratic_minimize(A, b, known=None):
    if known is None:
        if scipy.sparse.issparse(A):
            return scipy.sparse.linalg.spsolve(A.T@A, A.T@b)
        else:
            return scipy.linalg.solve(A.T@A, A.T@b)
    kid, kval = known
    mask = np.zeros(A.shape[1])
    mask[kid] = 1
    L = A.T@A
    rhs = A.T@b
    Lii = L[mask == 0][:, mask == 0]
    Lik = L[mask == 0][:, mask == 1]
    x = np.zeros((A.shape[1], 3))
    x[mask == 1] = kval
    x[mask == 0] = scipy.sparse.linalg.spsolve(Lii,
                                               rhs[mask == 0] - Lik@kval)
    return x


def quad_trim_assign(siblings, faces):
    quad_assign = - np.ones(len(faces), dtype=int)
    cnt = 0
    trimmers = np.zeros_like(faces, dtype=int)
    quads = []
    for i, s in enumerate(siblings):
        if s < 0:
            quad_assign[i] = -1
            continue
        if quad_assign[i] >= 0:
            assert quad_assign[s] >= 0
            continue
        fi, fs = set(faces[i]), set(faces[s])
        common = fi & fs
        assert len(common) == 2

        id_list = list(faces[i]) + list(fs-common)
        tid = list(faces[i]).index(list(fi-common)[0])
        permute = ([[0, 1, 3, 2], [0, 1, 2, 3], [0, 3, 1, 2]])[tid]
        q = np.array(id_list)[permute]

        id_map = {k: m for m, k in enumerate(q)}
        for ii in [i, s]:
            trimmers[ii] = [id_map[k] for k in faces[ii]]
        assert np.sum(q[trimmers[i]] - faces[i]) == 0
        quads.append(q)
        quad_assign[i] = quad_assign[s] = cnt
        cnt += 1
    assert trimmers.max() == 3
    assert quad_assign.max() * 2 < len(faces)

    q2t = - np.ones((len(quads), 2), dtype=int)
    for t, q in enumerate(quad_assign):
        if q < 0:
            continue
        if q2t[q][0] < 0:
            q2t[q][0] = t
        else:
            q2t[q][1] = t
    return quad_assign, q2t, trimmers, np.asarray(quads)


def tp_sample_value(x, y, order=3):
    tg1d = np.array([order - np.arange(order + 1),  np.arange(order + 1)]).T
    bas_x = feta.bernstein_evaluator(x, x*0, x*0, tg1d).T
    bas_y = feta.bernstein_evaluator(y, y*0, y*0, tg1d).T
    qt = quad_tuple_gen(order)
    bas_xy = np.array([bas_x[:, i]*bas_y[:, j] for i, j in qt]).T
    return bas_xy


def quad_tuple_gen(order):
    x, y = np.unravel_index(range((order+1)**2),
                            shape=(order+1, order+1))
    return np.vstack([x, y]).T


def sample_for_quad_trim(tr0, tr1, order: int):
    cache_key = (tuple(tr0), tuple(tr1), order)
    if cache_key in sample_for_quad_trim.cache:
        return sample_for_quad_trim.cache[cache_key]
    TG = np.asarray(feta.tuple_gen(order=order, var_n=2))
    V0 = TG@ref_quad[tr0].astype(int)
    V1 = TG@ref_quad[tr1].astype(int)
    bc = TG@ref_quad[[0, 1, 3]].astype(int)

    dim = (order+1, order+1)
    bc_collect = np.zeros(((order+1)**2,
                           4), dtype=int)
    # this implicitly correlate to the quad_tuple_gen which unravels.
    index0 = np.ravel_multi_index(V0.T, dims=dim)
    index1 = np.ravel_multi_index(V1.T, dims=dim)

    bc_collect[index0, 0] = 0
    bc_collect[index0, 2:] = bc
    bc_collect[index1, 0] = 1
    bc_collect[index1, 2:] = bc
    bc_collect[:, 1] = order - bc_collect[:, 2:].sum(axis=1)

    sample_for_quad_trim.cache[cache_key] = bc_collect
    return bc_collect


sample_for_quad_trim.cache = dict()


def standard_codec_list(level: int):
    std_x, std_y = quad_tuple_gen(level).T
    codes = []
    for (x, y) in zip(std_x, std_y):
        if x % level == 0 or y % level == 0:
            bc = [level - x, x, level - y, y]
            zid = bc.index(0)
            side = [[1, 2], [0, 3], [3, 2], [0, 1]][zid]
#             print(bc,zid,end=' ')
            bc = bc[2:] if zid <= 1 else bc[:2]
            code = tuple(side[:1]*bc[0] + side[1:]*bc[1])
#             print('/',x,y,code)
            codes.append(code)
        else:
            codes.append(tuple([-1]*level))
    return codes


def quad_ho_F(quads, level: int):
    std_cod = standard_codec_list(level)

    global_ids = np.zeros((len(quads), len(std_cod)), dtype=int)
    global_id_store = dict()
    cnt = 0
    for q in range(len(quads)):
        for i, c in enumerate(std_cod):
            assert len(c) == level
            if c[0] < 0:  # internal.
                node_id = cnt
                cnt += 1
            else:
                key = tuple(sorted(list(quads[q][cc] for cc in c)))
                if key not in global_id_store:
                    global_id_store[key] = cnt
                    cnt += 1
                node_id = global_id_store[key]
            global_ids[q, i] = node_id
    assert global_ids[0].max() == len(global_ids[0]) - 1
    assert global_ids.shape[1] == (level+1)**2
    return global_ids


def highorder_tp_sv(cp, level=3, order=3):
    def local_upsample(level: int):
        usV, usF = igl.upsample(np.array([[0, 0], [1, 0], [1, 1], [0, 1.]]),
                                np.array([[0, 1, 2], [0, 2, 3]]), level)
        bnd0 = igl.boundary_loop(usF)
        usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE
    usV, usF, usE = local_upsample(level=level)
    bas_uv = tp_sample_value(usV[:, 0], usV[:, 1], order)
    sv = bas_uv@cp.reshape(len(cp), -1, 3)
    return (sv,
            np.vstack([usF+i*len(usV) for i in range(len(sv))]),
            np.vstack([usE+i*len(usV) for i in range(len(sv))]))


def bezier_subd(cp):
    # de Casteljau
    l = [cp]
    for i in range(len(cp)-1):
        l.append((l[-1][1:] + l[-1][:-1])/2)
    return np.asarray([i[0] for i in l]), np.asarray([i[-1] for i in l[::-1]])


def bezier_eval(x, cp, order=3):
    tg1d = np.array([order - np.arange(order + 1),  np.arange(order + 1)]).T
    bas_x = feta.bernstein_evaluator(x, x*0, x*0, tg1d).T
    return bas_x@cp


def quad_fit(V, F, quads, q2t, trim_types, level, order, bsv, query):
    """Least Square Fitting for QuadMesh, with Tensor Product Bezier Basis

    Args:
        V ([type]): [description]
        F ([type]): [description]
        quads ([type]): [description]
        q2t ([type]): [description]
        trim_types (npt.Array): used for converting quad parameter values to triangle and barycentric coordinates.
        level ([type]): levels for upsample
        order ([type]): [description]
        bsv ([type]): [description]
        query (Callable): [description]

    Returns:
        [type]: [description]
    """
    F_qh_sample = quad_ho_F(quads, level=level)
    F_qh_order = quad_ho_F(quads, level=order)
    assert F_qh_sample.shape[1] == bsv.shape[0]

    all_samples = np.zeros((F_qh_sample.max() + 1, 3))
    ijk, vals = [], []

    for q, (t0, t1) in enumerate(q2t):
        tbc0 = np.array(sample_for_quad_trim(trim_types[t0], trim_types[t1], level),
                        dtype=int)
        tbc0[:, 0] = np.asarray([t0, t1])[tbc0[:, 0]]
        sample_vals = query(tbc0, denom=level)

        all_samples[F_qh_sample[q]] = sample_vals
        sample_q, order_q = F_qh_sample[q], F_qh_order[q]
        for i, si in enumerate(sample_q):
            for j, sj in enumerate(order_q):
                ijk.append((si, sj))
                vals.append(bsv[i, j])

    ijk, idx = np.unique(np.asarray(ijk), axis=0, return_index=True)

    A = scipy.sparse.csr_matrix(arg1=(np.array(vals)[idx],
                                      ijk.T))

    sol = quadratic_minimize(A, all_samples)

    all_cp = sol[F_qh_order]
    return all_cp


def solo_cc_split(V, F, siblings, t2q, quads, q_cp, order: int):
    """Catmul-Clark style split for the solo triangles.

    Returns information for constrained nodes and more.
    Args:
        V (np.array): TriMesh Vertices
        F (np.array): TriMesh Faces
        siblings ([type]): tris to sibling tris indices, only used -1 indicator here.
        t2q ([type]): tris to quad indices.
        quads ([type]): quad F matrix
        q_cp ([type]): control points for the existing quads.
        order (int): Polynomial order

    Returns:
        known_cp: a map from tuples to control points. Used for subsequent fitting.
    """
    assert len(q_cp) == len(quads)
    solos = np.where(siblings == -1)[0]

    TT, TTi = igl.triangle_triangle_adjacency(F)

    edge_node_map = {tuple(sorted(k)): i
                     for i, k in enumerate(standard_codec_list(order))}
    edge_id = - np.ones_like(TT)
    avail_id = F.max() + 1
    known_cod2cp = dict()
    newverts = []
    for f in solos:
        for e in range(3):
            v0, v1 = F[f, e], F[f, (e+1) % 3]
            fo, eo = TT[f, e], TTi[f, e]
            if fo < 0:  # boundary, future more can go here
                edge_id[f, e] = avail_id
                newverts.append((V[v0]+V[v1])/2)
                avail_id += 1
                continue
            qid = t2q[fo]
            if qid < 0:  # not a quad, this would be activated later
                if edge_id[f, e] < 0:
                    edge_id[f, e] = edge_id[fo, eo] = avail_id
                    newverts.append((V[v0]+V[v1])/2)
                    avail_id += 1
                continue
            assert edge_id[f, e] < 0, "Could not have visited twice."
            quad = list(quads[qid])

            qv0, qv1 = [quad.index(v0), quad.index(v1)]
            # collect on this edge
            tup1d = np.array([order - np.arange(order + 1),
                              np.arange(order + 1)]).T

            tup_list = [tuple(sorted([qv0]*t0 + [qv1]*t1)) for t0, t1 in tup1d]
            cp_list = np.asarray([q_cp[qid][edge_node_map[t]]
                                  for t in tup_list])
            cp0, cp1 = bezier_subd(cp_list)
            for i, (t0, t1) in enumerate(tup1d):
                known_cod2cp[tuple(sorted([v0]*t0 + [avail_id]*t1))] = cp0[i]
                known_cod2cp[tuple(sorted([avail_id]*t0 + [v1]*t1))] = cp1[i]

            edge_id[f, e] = avail_id
            newverts.append((V[v0]+V[v1])/2)
            avail_id += 1
    newquads = []
    for pid, fi in enumerate(solos, avail_id):  # new internal id
        v0, v1, v2 = F[fi]
        for ei in range(3):
            v0, e0, e2 = F[fi, ei], edge_id[fi, ei], edge_id[fi, (ei+2) % 3]
            newquads += [[v0, e0, pid, e2]]
    assert np.asarray(newquads).max() + 1 == avail_id + len(solos)
    return (np.concatenate([newverts, igl.barycenter(V, F[solos])]),
            known_cod2cp, newquads)


def constrained_cc_fit(V, F, siblings, newquads, known_cp, level, order, bsv, query):
    solos = np.where(siblings == -1)[0]
    newF = F[siblings == -1]
    assert len(newF)*3 == len(newquads)
    F_sa = quad_ho_F(newquads, level=level)
    F_or = quad_ho_F(newquads, level=order)

    std_codec = standard_codec_list(order)
    known_dict = dict()
    for q, quad in enumerate(newquads):
        for e, code in enumerate(std_codec):
            if code[0] < 0:
                continue
            key = tuple(sorted(list(quad[cc] for cc in code)))
            if key in known_cp:
                if F_or[q, e] in known_dict:
                    assert (known_dict[F_or[q, e]] == known_cp[key]).all()
                else:
                    known_dict[F_or[q, e]] = known_cp[key]

    known_ids = np.ones(len(known_dict), dtype=int)
    known_vals = np.ones((len(known_dict), 3))
    for i, (k, v) in enumerate(known_dict.items()):
        known_ids[i], known_vals[i] = k, v

    all_samples = np.zeros((F_sa.max() + 1, 3))
    ijk, vals = [], []

    refpts = np.array([[0, 0],
                       [3, 0],
                       [6, 0],
                       [3, 3],
                       [0, 6],
                       [0, 3],
                       [2, 2]])  # denom=6
    cc_q = np.array([[0, 1, 6, 5], [2, 3, 6, 1], [4, 5, 6, 3]])
    type2bc = []
    for t in range(3):
        x, y = np.array(quad_tuple_gen(level)).T
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        assert x.max() <= level
        qc = refpts[cc_q[t]].reshape(-1, 1, 2)
        type2bc.append(((level-x)*qc[0] + x*qc[1])*(level-y) +
                       ((level-x)*qc[3] + x*qc[2])*y)
    for qi, q in enumerate(newquads):
        tbc = np.zeros(((level+1)**2, 4),
                       dtype=int)
        tbc[:, 0] = solos[qi//3]
        tbc[:, 2:] = type2bc[qi % 3]
        tbc[:, 1] = 6*(level)**2 - tbc[:, 2:].sum(axis=1)
        assert tbc[:, 1].min() >= 0
        sample_vals = query(tbc, denom=6*(level)**2)
        all_samples[F_sa[qi]] = sample_vals
        sample_q, order_q = F_sa[qi], F_or[qi]
        for i, si in enumerate(sample_q):
            for j, sj in enumerate(order_q):
                ijk.append((si, sj))
                vals.append(bsv[i, j])

    ijk, idx = np.unique(np.asarray(ijk), axis=0, return_index=True)
    A = scipy.sparse.csr_matrix(arg1=(np.array(vals)[idx],
                                      ijk.T))
    sol = quadratic_minimize(A, all_samples, (known_ids, known_vals))
    all_cp = sol[F_or]
    return all_cp
