### Scripts for generating figures in the Shell paper.

from vis_utils import *

def heat_create():
    refV, refF, B, T, V, F, _ = h5deser(
        '/home/zhongshi/Workspace/nutshell_data/raw_data/121868.stl.h5')

    heat = seism.HeatGeodesicsData()
    heat.precompute(secV, secF, False)  # np.linalg.norm(secV[314] - refV[114])
    sec_d = heat.solve(np.array([314]))

    with h5py.File('../buildr/heat_xf.h5', 'w') as fp:
        fp['mV'], fp['mF'], fp['pV'], fp['pF'] = refV, refF, secV, secF

    with h5py.File('../buildr/heat_xf.h5', 'r') as fp:
        fid, qUV = fp['fid'][()], fp['qUV'][()]

    trans_d = np.sum(
        sec_d[secF[fid]]*np.hstack([1-qUV.sum(axis=1)[:, None], qUV]), axis=1)

    heat = seism.HeatGeodesicsData()
    heat.precompute(refV, refF, True)
    intri_d = heat.solve(np.array([114]))

    iV0, iE0 = igl.isolines(refV, refF, d, 20)
    iV1, iE1 = igl.isolines(refV, refF, in_d, 20)

    exact_d = igl.exact_geodesic(refV, refF.astype(np.long),
                                 np.array([114]),
                                 np.arange(len(refV)))

    from matplotlib import cm

    ply_writer('/home/zhongshi/Workspace/nutshell_data/heat_in_shell/ref.ply',
               refV, refF, (cm.viridis(exact_d)[:, :3]*255).astype(np.uint))

    igl.write_obj('/home/zhongshi/Workspace/nutshell_data/heat_in_shell/ref_isoline.obj', iV0,
                  np.hstack([iE0, np.zeros((len(iE0), 1))]).astype(np.int))


def vary_thick():
    scale(refV)
    for i, thi in enumerate(['0.5', '0.1', '0.01', '0.001']):
        refV, refF, B, T, V, F, _ = h5deser(
            f'/home/zhongshi/Workspace/nutshell_data/raw_data/78481.stl.h5.{thi}.init.h5')
        T = use_scale(T)
        F = F[T[F, 1].mean(axis=1) > 0]
        igl.write_triangle_mesh(
            f'/home/zhongshi/Workspace/nutshell_data/vary_thick/{i}.obj', T, F)
    igl.write_triangle_mesh(
        '/home/zhongshi/Workspace/nutshell_data/vary_thick/ref.obj', use_scale(refV), refF)
    heights = []
    # statistics
    refV, refF, sB, sT, sV, sF, _ = h5deser(
        '../../nutshell_data/raw_data/78481.stl.h5.0.001.init.h5')
    heights.append(sT-sB)  # F.shape,refF.shape)
    refV, refF, sB, sT, sV, sF, _ = h5deser(
        '../../nutshell_data/raw_data/78481.stl.h5.0.01.init.h5')
    heights.append(sT-sB)  # F.shape,refF.shape)
    refV, refF, sB, sT, sV, sF, _ = h5deser(
        '../../nutshell_data/raw_data/78481.stl.h5.0.1.init.h5')
    heights.append(sT-sB)  # F.shape,refF.shape)
    refV, refF, sB, sT, sV, sF, _ = h5deser(
        '../../nutshell_data/raw_data/78481.stl.h5.0.5.init.h5')
    heights.append(sT-sB)  # F.shape,refF.shape)

    go.Figure([go.Box(y=(np.linalg.norm(h, axis=1))) for h in heights[::-1]]).update_layout(yaxis_type='log').update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                                                                                                            plot_bgcolor='rgba(0,0,0,0)',
                                                                                                                            yaxis=dict(
                                                                                                                                gridcolor='grey', gridwidth=0.5, zeroline=True),
                                                                                                                            geo=dict(showframe=True)).write_image(
        'vary_thick.svg', height=435, width=2000)


def visualize_tetmesh():
    refV, refF, B, T, V, F, _ = h5deser(
        '/home/zhongshi/Workspace/nutshell_data/raw_data/knot100K.obj.h5')
    secV, secF = igl.read_triangle_mesh(
        '/home/zhongshi/Workspace/nutshell_data/raw_data/knot100K.obj.h5.ply')
    import pymesh

    colorA = np.array([192, 192, 192])/256
    colorB = np.array([249, 161, 94])/256

    mesh = pymesh.load_mesh('simple_knot.msh')

    def clip_and_color(mesh):
        x, y, z = mesh.vertices[mesh.voxels].mean(axis=1).T
        clip_index = np.where(0.5*x + 0.5*z - 0.48 >= 0)[0]

        fc = igl.boundary_facets(mesh.voxels[clip_index])

        original_faces = set(frozenset(f) for f in mesh.faces)
        from_facet = [i for i, f in enumerate(
            fc) if frozenset(f) in original_faces]
        return mesh.vertices, fc, from_facet

    V, F, facet = clip_and_color(mesh)
    meshraw = pymesh.load_mesh('raw_knot.msh')

    rV, rF, rfacet = clip_and_color(meshraw)

    full_color = np.tile(colorA, (F.shape[0], 1))
    full_color[facet] = colorB
    rfull_color = np.tile(colorA, (rF.shape[0], 1))
    rfull_color[rfacet] = colorB

    plt = None
    sh = dict(width=1000, height=1000, wireframe=True)
    vw = mp.Viewer(sh)
    vw.add_mesh(V, F, c=full_color, shading=sh)
    plt = mp.Subplot(plt, vw, s=[1, 2, 1])
    vw = mp.Viewer(sh)
    vw.add_mesh(rV, rF, c=rfull_color, shading=sh)
    plt = mp.Subplot(plt, vw, s=[1, 2, 0])


def armadillo_cages():
    refV, refF, B, T, V, F, _ = h5parse(f'../buildr/armadillo.obj.h5')

    top = [T]
    faces = [F]
    for i in range(3):
        _, _, B, T, V, F, _ = h5parse(f'../buildr/{i}blow.h5')
        top.append(T)
        # bases.append(B)
        faces.append(F)

    colors = [
        np.array([0.4, 0.4, 0.4]),
        np.array([100., 100, 170])/256,
        np.array([202., 47, 80])/256,
        np.array([221., 160, 211])/256
    ]
    plt = mplot(refV, refF, c=colors[0],
                wireframe=False, width=1000, height=1000)
    V = top[0]
    F = np.array([f for f in faces[0] if V[f].sum() > -1.5])
    plt.add_mesh(V, F, c=colors[1], shading=dict(line_color='red'))
    V = top[1]
    F = np.array([f for f in faces[1] if V[f].sum() > 0.5])
    plt.add_mesh(V, F, c=colors[2], shading=dict(line_color='green'))
    V = top[2]
    F = np.array([f for f in faces[2] if V[f].sum() > 2.5])
    plt.add_mesh(V, F, c=colors[3], shading=dict(line_color='green'))


def wheel_setup():
    refV, refF, sB, sT, sV, sF, trackers = h5deser('../buildr/1517923.obj.h5')

    secV, secF = igl.read_triangle_mesh('../buildr/1517923.obj.h5.ply')

    mesh = pymesh.load_mesh('wheel.mesh')

    bc = mesh.vertices[mesh.faces].mean(axis=1)

    secV, SVI, SVJ, _ = igl.remove_unreferenced(mesh.vertices, mesh.faces)
    secF = SVJ[mesh.faces]

    plt = mplot(refV, (refF), wireframe=False)
    plt.add_points(np.array(_111).T, shading=dict(ponit_size=1))

    with h5py.File('../buildr/wheelXf.h5', 'w') as fp:
        fp['pV'], fp['pF'], fp['mV'], fp['mF'] = secV, secF, refV, refF

    with h5py.File('../buildr/wheelXf.h5', 'r') as fp:
        fid, qUV = fp['fid'][()], fp['qUV'][()]

    np.vstack([(refV[refF[fid], i]*np.hstack([1-qUV.sum(axis=1, keepdims=True),
                                              qUV[:, :1], qUV[:, 1:]])).sum(axis=1) for i in range(3)]).T

    colors = np.zeros((len(mesh.faces)))
    colors[np.abs((bc[:, 0]-0.5)**2 + (bc[:, 1]-0.5)**2) < 0.15**2] = 1
    colors[np.abs((bc[:, 0]-0.935)**2 + (bc[:, 1]-0.5)**2) < 0.045**2] = 2
    colors[np.abs((bc[:, 0]-0.065)**2 + (bc[:, 1]-0.5)**2) < 0.045**2] = 3

    plt = mplot(mesh.vertices, mesh.faces, c=colors)
    # plt.add_edges(sT,sF)


def wheelfem():
    with np.load('~/Workspace/nutshell_data/teaser/save.npz') as npl:
        p, t, d, m = npl['p'], npl['t'], npl['d'], npl['m']
        #print([k for k in npl.keys()])
    with np.load('/home/zhongshi/Workspace/nutshell_data/teaser/save.npz') as npl:
        p, t, d, m = npl['p'], npl['t'], npl['d'], npl['m']
        print([l for l in npl.keys()])

    SV, SVI, SVJ, _ = igl.remove_duplicate_vertices(
        p, np.array([[1, 2, 3]]), 1e-10)
    tree = scipy.spatial.cKDTree(SV)
    _, ind = tree.query(tV)
    colors = cm.cividis(np.clip(m[SVI][ind].flatten(), 0, 1e5)/1e5)[:, :3]
    mplot(d[SVI][ind]+SV[ind], tF, c=colors, wireframe=False)


def draw_wheel():
    _, _, sB, sT, sV, sF, _ = h5parse('../tests/data/1517923.obj.h5')

    tV, tD, mV, tF, mF = h5reader(
        '../tests/data/wheel_stretch.h5', ['tV', 'tD', 'mV', 'tF', 'mF'])
    fV, fF = igl.read_triangle_mesh('../tests/data/wheel_transfered.obj')

    plt = None
    sh = dict(width=1200, height=1000, wireframe=True,
              wire_color='rgb(20,20,20)')
    vw = mp.Viewer(sh)
    vw.add_mesh(mV, mF, c=np.array([100, 100, 170])/256, shading=sh)
    # vw.add_edges(*igl.isolines(tV,tF,D0,10))
    plt = mp.Subplot(plt, vw, s=[2, 2, 0])

    vw = mp.Viewer(sh)
    vw.add_mesh(V, F, c=np.array([100, 100, 170])/256, shading=sh)
    vw.add_edges(sT, f2e(sF), shading=dict(
        line_width=2, line_color='rgb(156, 159, 157)'))
    plt = mp.Subplot(plt, vw, s=[2, 2, 2])

    vw = mp.Viewer(sh)
    vw.add_mesh(tV+tD, tF, c=np.array([100, 100, 170])/256, shading=sh)
    plt = mp.Subplot(plt, vw, s=[2, 2, 3])

    vw = mp.Viewer(sh)
    vw.add_mesh(fV, fF, c=np.array([100, 100, 170])/256, shading=sh)
    # vw.add_edges(*igl.isolines(V0,F0,transfered,10))
    plt = mp.Subplot(plt, vw, s=[2, 2, 1])


def intersect_leg_process():
    refV, refF, sB, sT, sV, sF, trackers = h5deser(
        '/home/zhongshi/Workspace/nutshell_data/raw_data/leg-intersect.off.h5')

    import sys
    sys.path.append('../python')
    import seism

    stdV, stdF = igl.upsample(
        np.array([[0, 0], [1, 0], [0, 1.]]), np.array([[0, 1, 2]]), 4)

    u, v = stdV[:, :1], stdV[:, 1:]
    allV = np.einsum('fjk,bj->fbk', sV[sF],
                     np.hstack([1-u-v, u, v])).reshape(-1, 3)
    allF = np.vstack([stdF+len(u)*i for i in range(len(sF))])

    allB = np.einsum('fjk,bj->fbk', sB[sF],
                     np.hstack([1-u-v, u, v])).reshape(-1, 3)
    allT = np.einsum('fjk,bj->fbk', sT[sF],
                     np.hstack([1-u-v, u, v])).reshape(-1, 3)

    secV, SVI, SVJ, _ = igl.remove_duplicate_vertices(allV, allF, 1e-10)
    secF = SVJ[allF]
    # c=(np.arange(len(secF))//16)
    # secV == SVI[allV]

    secB, secT = allB[SVI], allT[SVI]

    v_parent_f = np.repeat(np.arange(len(sF)), len(u))[SVI]

    import sys
    results = -np.ones((len(secB), 4))
    for fi in tqdm.trange(len(trackers)):  # for each coarse face
    track_f = trackers[fi]  # find and slice it.
    lV, lF, _, _ = igl.remove_unreferenced(refV, refF[track_f])
    if len(track_f) == 1:
        lF = np.array([lF])
    assert len(lF) == len(trackers[fi])
    tree = seism.AABB(lV, lF)
    v_to_consider = np.where(v_parent_f == fi)[0]
    for v in v_to_consider:
        sys.stdout.flush()
        hf, hu, hv, ht = (tree.segment_hit(secB[v], secT[v], False))
        results[v] = np.array([track_f[hf], 1-hu-hv, hu, hv])


def bunny_disp():
    import scipy
    import scipy.interpolate
    with h5py.File('../buildr/bunny_uvt.h5', 'r') as fp:
        uvtn, idptr = fp['uvt'][()], fp['prism_ptr'][()]

    refV, refF, base, top, V, F, _ = h5deser('../buildr/bunny.off.h5')
    direction = top-base
    usV, usF = igl.upsample(
        np.array([[0, 0.], [1, 0], [0, 1]]), np.array([[0, 1, 2]]), 3)
    V_list, F_list = [], []
    u, v = usV.T
    u, v = u[:, None], v[:, None]
    v_cnt = 0
    for i, f in enumerate(F):
        x, y, t, px, py, pz = uvtn[idptr[i]:idptr[i+1]].T
        interx = scipy.interpolate.NearestNDInterpolator(
            np.vstack([x, y]).T, px)
        intery = scipy.interpolate.NearestNDInterpolator(
            np.vstack([x, y]).T, py)
        interz = scipy.interpolate.NearestNDInterpolator(
            np.vstack([x, y]).T, pz)
        newPoints = np.vstack([interx(usV).flatten(), intery(
            usV).flatten(), interz(usV).flatten()]).T

        V_list.append(newPoints)
        F_list.append(usF + v_cnt)
        v_cnt += len(u)

    V_combine, F_combine = np.vstack(V_list), np.vstack(F_list)
    newV, SVi, SVJ, _ = igl.remove_duplicate_vertices(
        V_combine, F_combine, 1e-15)
    newF = SVJ[F_combine]


def boolean_postmesh():
    mesh = pymesh.load_mesh('../tests/data/birdengine_boolean.ply')

    refV, refF, B, T, V, F, _ = h5deser(
        '/home/zhongshi/Workspace/nutshell_data/raw_data//birdengine_boolean.ply.h5')

    pV, pF = igl.read_triangle_mesh('../buildr/birdengine_boolean.ply.h5.ply')
    uV, uF = igl.upsample(pV, pF, 4)

    with h5py.File('../buildr/csgXf.h5', 'w') as fp:
        fp['pV'], fp['pF'], fp['mV'], fp['mF'] = uV, uF, refV, refF

    with h5py.File('../buildr/csgXf.h5', 'r') as fp:
        u4fid, u4qUV = fp['fid'][()], fp['qUV'][()]

    colors = mesh.get_attribute('face_source')[u4fid]
    transition_faces = np.where(colors[uF].sum(axis=1) % 3 > 0)[0]
    transition_verts = uV[np.array(uF[transition_faces]).flatten()]
    pmesh = pymesh.form_mesh(pV, pF)
    _, pfid, _ = pymesh.distance_to_mesh(pmesh, transition_verts)

    pVlist = set(pF[np.unique(pfid)].flatten())

    stripe_id = [i for i, f in enumerate(pF) if len(set(f) & pVlist) > 0]

    rV, rF, _, _ = igl.remove_unreferenced(pV, pF[np.unique(stripe_id)])
    rV, rF = igl.upsample(rV, rF, 6)
    with h5py.File('../buildr/csg_strip.h5', 'w') as fp:
        fp['pV'], fp['pF'], fp['mV'], fp['mF'] = rV, rF, refV, refF

    with h5py.File('../buildr/csg_strip.h5', 'r') as fp:
        fid, qUV = fp['fid'][()], fp['qUV'][()]

    invert_id = np.array([i for i in range(len(pF)) if i not in stripe_id])

    allcolor = [[231, 167, 38]if i == 1 else [210, 210, 210] for i in np.concatenate(
        [mesh.get_attribute('face_source')[u4fid[:len(pV)]], mesh.get_attribute('face_source')[fid]])]
    ply_writer('colored_bird.ply', np.vstack([pV, rV]), np.vstack(
        [pF[invert_id], rF+len(pV)]), allcolor)


def inverse_phong_projection(refV,refF,V1,F1, VN1):
    from scipy.optimize import least_squares

    def phong_coeffs(V0, V1, V2, N0, N1, N2, Q):
        return (np.cross(V1-V0, N1-N0),  # a2
                np.cross(V2-V0, N2-N0),  # b2
                np.cross(V1-V0, N2-N0)+np.cross(V2-V0, N1-N0),  # ab
                np.cross(V0-Q, N1-N0)+np.cross(V1-V0, N0),  # a
                np.cross(V2-V0, N0) + np.cross(V0-Q, N2-N0),  # b
                np.cross(V0-Q, N0))

    queryV = np.zeros_like(refV)
    for vid, v in enumerate(tqdm.tqdm(refV)):
        for f in F1: # requires optimization with inversed `track` if to be seriously used.
            coef = phong_coeffs(*V1[f], *VN1[f], v)

            def quadr(x):
                return sum(coef[k]*x[0]**i*x[1]**j for k, (i, j) in enumerate([(2, 0), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)]))
            sol = least_squares(quadr, np.array([1/3, 1/3]), bounds=(0, 1))
            x = sol.x
            if sol.cost <= 1e-8 and x.sum() <= 1:
                # print(sol)
                queryV[vid] = (1-x.sum())*V1[f[0]] + \
                    x[0]*V1[f[1]]+x[1]*V1[f[2]]
                # print('f:',f)
                break


def shark():
    refV, refF, B, T, V, F, _ = h5deser('../buildr/scalability/shark.obj.h5')

    import sys
    sys.path.append('../python/')
    import seism

    pc = seism.PrismCage('../buildr/scalability/shark.obj.h5')
    fid, quv = pc.transfer(V, F, refV)

    pts = np.einsum('vji,vj->vi', V[F[fid]],
                    np.hstack([1-quv.sum(axis=1)[:, None], quv]))

    fid2, quv2 = pc.transfer(refV, refF, pts)

    pts2 = np.einsum(
        'vji,vj->vi', refV[refF[fid2]], np.hstack([1-quv2.sum(axis=1)[:, None], quv2]))

    errors = np.linalg.norm(pts2-refV, axis=1)

    queryV = np.load('queryV.npy')
    _, V1, F1, _, _ = igl.qslim(refV, refF, len(F))

    VN1 = igl.per_vertex_normals(V1, F1)

    qf, qu, qv = queryV.T
    qpts = np.einsum(
        'vji,vj->vi', V1[F1[qf.astype(np.int)]], np.vstack([1-qu-qv, qu, qv]).T)

    qnormal = np.einsum(
        'vji,vj->vi', VN1[F1[qf.astype(np.int)]], np.vstack([1-qu-qv, qu, qv]).T)

    import tqdm
    aabb = seism.AABB(refV, refF)
    queries = -np.ones((len(qpts), 3))
    for i, (v, n) in enumerate(zip(tqdm.tqdm(qpts), qnormal)):
        side1 = aabb.segment_hit(v, v+n, True)
        side2 = aabb.segment_hit(v, v-n, True)
        if side1[0] == -1 or side1[-1] > side2[-1]:
            side1 = side2
        queries[i] = side1[:-1]

    qf2, qu2, qv2 = queries.T
    qpts2 = np.einsum(
        'vji,vj->vi', refV[refF[qf2.astype(np.int)]], np.vstack([1-qu2-qv2, qu2, qv2]).T)
    go.Figure([go.Box(x=np.linalg.norm(qpts2-refV, axis=1)), go.Box(x=errors)],
              layout=dict(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          yaxis=dict(gridcolor='grey',
                                     gridwidth=0.5, zeroline=True),
                          geo=dict(showframe=True), showlegend=False,
                          font=dict(family='Linux Libertine'))).update_layout(xaxis_type='log', xaxis=dict(showexponent='all', exponentformat='e'))

    
def hairy_bunny():
    refV,refF,B,T,V,F,_ = h5deser('../buildr/bunny.off.h5')

    uB, _ = igl.upsample(B,F,3)
    uT, _ = igl.upsample(T,F,3)

    import tqdm
    import pymesh
    all_v = []
    all_f = []
    all_c = []
    curv_v = 0
    curv_col = 0
    for b, t in zip(tqdm.tqdm(uB),uT):
        m = pymesh.generate_cylinder(b,t, 1e-3,6e-4,6)
        all_v.append(m.vertices)
        all_f.append(m.faces + curv_v)
        all_c.append(np.ones(len(m.faces))*(b[:2]))
        curv_v += m.vertices.shape[0]
        curv_col += 1

    with Sidecar():
        mplot(np.vstack(all_v), np.vstack(all_f), 
              c = np.hstack(all_c),wireframe=False,
             width=1000,height=1000)

    np.savez('hairy_bunny.npz', 
             uB=uB,
             uT=uT, all_v=all_v,all_f=all_f)

    with np.load('hairy_bunny.npz') as npl:
        uB, uT, all_v, all_f = npl['uB'], npl['uT'], npl['all_v'], npl['all_f']

    import sys

    sys.path.append('../python')

    import seism

    aabb = seism.AABB(refV,refF)

    import tqdm
    all_hits = []
    for b, t in zip(tqdm.tqdm(uB),uT):
        all_hits.append(aabb.segment_query(b,t))
        #break

    V_hit = np.array(all_hits)

    # all_c = []
    # for b, t in zip(tqdm.tqdm(uB),uT):
    #     all_c.append(np.ones(24)*(b[1]+b[0]))
    with Sidecar():
        plt = mplot(np.vstack(all_v), np.vstack(all_f), 
                    c = hitcol[:,:3], #np.repeat(V_hit[:,1]+V_hit[:,0],(24)),
                    wireframe=False,
                    width=1000,height=1000,
                    )
        plt.add_mesh(refV,refF, c=refcol[:,:3],
                     shading=dict(colormap='RdYlGn',flat=False))

    with Sidecar():
        mplot(refV,refF,c=refV[:,1])

    ply_writer('hairs.ply',np.vstack(all_v), np.vstack(all_f), (hitcol[:,:3]*255).astype(np.uint))

    ply_writer('bunny_h.ply',refV,refF, (refcol[:,:3]*255).astype(np.uint))

    refcol = cm.RdYlGn_r((refV[:,1]+refV[:,0]-0.15)/1.4)

    hitcol = cm.RdYlGn_r(np.repeat((V_hit[:,1]+V_hit[:,0]-0.15)/1.4,
                         14))