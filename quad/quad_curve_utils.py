
def quad_trim_assign(siblings, faces):
    quad_assign = - np.ones(len(faces),dtype=int)
    cnt = 0
    trimmers = np.zeros_like(faces, dtype=int)
    quads = []
    for i,s in enumerate(siblings):
        if s < 0:
            quad_assign[i] = -1
            continue
        if quad_assign[i] >= 0:
            assert quad_assign[s] >=0
            continue
        fi, fs = set(faces[i]), set(faces[s])
        common = fi & fs
        assert len(common) == 2
        
        id_list = list(faces[i]) + list(fs-common)
        tid = list(faces[i]).index(list(fi-common)[0])
        permute = ([[0,1,3,2], [0,1,2,3],[0,3,1,2]])[tid]
        q = np.array(id_list)[permute]
        
        id_map = {k:m for m,k in enumerate(q)}
        for ii in [i,s]:
            trimmers[ii] = [id_map[k] for k in faces[ii]]
        assert np.sum(q[trimmers[i]] - faces[i]) == 0
        quads.append(q)
        quad_assign[i] = quad_assign[s] = cnt
        logger.debug([faces[i], faces[s],id_list, tid, 
                      'trim',trimmers[i], trimmers[s]])
        cnt += 1
    assert trimmers.max() == 3
    assert quad_assign.max()* 2 < len(faces)
    
    q2t = - np.ones((len(quads),2), dtype=int)
    for t,q in enumerate(quad_assign):
        if q < 0: continue
        if q2t[q][0] < 0:
            q2t[q][0] = t
        else:
            q2t[q][1] = t
    return quad_assign, q2t, trimmers, np.asarray(quads)


def tp_sample_value(x,y, order = 3):
    tg1d = np.array([order - np.arange(order + 1),  np.arange(order + 1)]).T
    bas_x = feta.bernstein_evaluator(x,x*0,x*0, tg1d).T
    bas_y = feta.bernstein_evaluator(y,y*0,y*0, tg1d).T
    qt = quad_tuple_gen(order)
    bas_xy = np.array([bas_x[:, i]*bas_y[:, j] for i,j in qt]).T
    return bas_xy

def quad_tuple_gen(order):
    x,y = np.unravel_index(range((order+1)**2), 
                            shape=(order+1,order+1))
    return np.vstack([x,y]).T

def sample_for_quad_trim(tr0, tr1, order:int):
    cache_key = (tuple(tr0), tuple(tr1), order)
    if cache_key in sample_for_quad_trim.cache:
        return sample_for_quad_trim.cache[cache_key]
    TG = np.asarray(feta.tuple_gen(order=order,var_n=2)) 
    V0 = TG@ref_quad[tr0].astype(int)
    V1 = TG@ref_quad[tr1].astype(int)
    bc = TG@ref_quad[[0,1,3]].astype(int)
    
    dim = (order+1, order+1)
    bc_collect = np.zeros(((order+1)**2,
                           4),dtype=int)
    index0 = np.ravel_multi_index(V0.T, dims=dim)
    index1 = np.ravel_multi_index(V1.T, dims=dim)
    
    bc_collect[index0,0] = 0
    bc_collect[index0, 2:] = bc
    bc_collect[index1, 0] = 1
    bc_collect[index1,2:] = bc
    bc_collect[:, 1] = order - bc_collect[:,2:].sum(axis=1)
    
    sample_for_quad_trim.cache[cache_key] = bc_collect
    return bc_collect
sample_for_quad_trim.cache = dict()


def standard_codec_list(level: int):
    std_x, std_y = quad_tuple_gen(level).T
    codes = []
    for (x,y) in zip(std_x, std_y):
        if x % level == 0 or y % level == 0:
            bc = [level - x, x, level - y, y]
            zid = bc.index(0)
            side = [[1,2],[0,3],[3,2],[0,1]][zid]
#             print(bc,zid,end=' ')
            bc = bc[2:] if zid <= 1 else bc[:2]
            code = tuple(side[:1]*bc[0] + side[1:]*bc[1])
#             print('/',x,y,code)
            codes.append(code)
        else:
            codes.append(tuple([-1]*level))
    return codes



def quad_ho_F(q2t, quads, trim_types, level:int):
    std_cod = standard_codec_list(level)

    global_ids = np.zeros((len(q2t), len(std_cod)), dtype=int)
    global_id_store = dict()
    cnt = 0
    for q, (t0,t1) in enumerate(q2t):
        for i,c in enumerate(std_cod):
            assert len(c) == level
            if c[0] < 0:
                node_id = cnt
                cnt += 1
            else:
                key = tuple(sorted(list(quads[q][cc] for cc in c)))
                if key not in global_id_store:
                    global_id_store[key] = cnt
                    cnt += 1
                node_id = global_id_store[key]
            global_ids[q,i] = node_id
    assert global_ids[0].max() == len(global_ids[0]) - 1
    assert global_ids.shape[1] == (level+1)**2
    return global_ids

from curve import fem_generator as feta
def highorder_tp_sv(cp, level=3, order=3):
    def local_upsample(level:int):
        usV, usF = igl.upsample(np.array([[0,0],[1,0],[1,1],[0,1.]]), 
                                np.array([[0,1,2],[0,2,3]]), level)
        bnd0 = igl.boundary_loop(usF)
        usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE
    usV,usF,usE = local_upsample(level=level)
    bas_uv = tp_sample_value(usV[:,0],usV[:,1], order)
    sv = bas_uv@cp.reshape(-1,3)
    return (sv, 
            np.vstack([usF+i*len(usV) for i in range(len(sv))]), 
            np.vstack([usE+i*len(usV) for i in range(len(sv))]))
