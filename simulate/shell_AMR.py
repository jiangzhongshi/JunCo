import triangle_rolls
import subprocess
import tube_gen
import numpy as np
import meshplot as mp
import meshio
import igl
import re
import glob


def explode_uv_coord(verts, faces):
    """convert 3d mesh of a cylinder to UV coords.

    Args:
        verts ([type]): [description]
        faces ([type]): [description]

    """
    theta = np.arctan2(verts[:, 1], verts[:, 2])/(2*np.pi)  # [-0.5, 0.5]
    theta[np.abs(theta) < 1e-3] = 0
    uv = np.stack((verts[:, 0], theta)).T

    bc = igl.barycenter(uv, faces)
    explode = uv[faces]

    for i, f in enumerate(faces):
        down = explode[i, :, 1].min() < 0
        for j in range(3):
            if explode[i, j, 1] < 0 or (down and explode[i, j, 1] == 0):
                explode[i, j, 1] += 1
    v_range = np.arange(len(verts))
    exp_v, exp_f = explode.reshape(-1,
                                   2), np.arange(len(faces)*3).reshape(-1, 3)
    _, uind, uinv = np.unique(np.round(
        exp_v*1e8).astype(int), axis=0, return_index=True, return_inverse=True)
    expand_3d = np.zeros((len(uind), 3))
    expand_3d[:, :2] = exp_v[uind]
    return expand_3d, uinv[exp_f], v_range[faces].flatten()[uind]


def transfer_pymesh(src_v, src_f, src_u, dense_v, dense_f):
    """Use PyMesh AABB closest point to transfer attributes from src to dense.

    Args:
        src_v ([type]): [description]
        src_f ([type]): [description]
        src_u ([type]): [description]
        dense_v ([type]): [description]
        dense_f ([type]): [description]

    Returns:
        [type]: [description]
    """
    import pymesh
    pm1 = pymesh.form_mesh(src_v, src_f)
    pm0 = pymesh.form_mesh(dense_v, dense_f)
    pm1.add_attribute('disp')
    pm1.set_attribute('disp', src_u)  # u1[uind][ptr])
    pymesh.map_vertex_attribute(pm1, pm0, 'disp')
    return pm0.get_attribute('disp').reshape(-1, 3)


def load_and_process(name):
    '''
    Peel the outer layer of a shell cylinder, and unwrap to UV for cylinder.
    '''
    m1 = meshio.read(name)
    v1, u1, f1 = m1.points, m1.point_data['solution'], m1.cells[0][1]

    _, uind, uinv = np.unique(np.round(v1 * 1e7).astype(int), return_index=True, return_inverse=True,
                              axis=0)

    new_v, new_f = v1[uind], igl.boundary_facets(uinv[f1])

    new_sol = u1[uind]

    face_normal = igl.per_face_normals(new_v, new_f, np.ones(3))
    bc = igl.barycenter(new_v, new_f)

    outer_faces = new_f[np.sum(face_normal[:, 1:]*bc[:, 1:], axis=1) > 1e-2]

    exp_v, exp_f, ptr = explode_uv_coord(new_v, outer_faces)
    return exp_v, exp_f, new_sol[ptr], new_v[ptr]


def expand_color(uv0, tf0, faces_markers):
    """slightly expand the face markers by one-ring"""
    v_col = np.zeros(len(uv0))
    v_col[tf0[faces_markers].flatten()] = 1
    expand_col = np.zeros(len(tf0))
    for i, f in enumerate(tf0):
        if v_col[f].max() > 0:
            expand_col[i] = 1
    return np.where(expand_col > 0)[0]


def look_up(tv, tf, d_v, sol_mark, expand=False):
    """uv lookup where the solution is exceeding."""
    import pymesh
    tree = pymesh.AABBTree()
    tv3 = np.zeros((len(tv), 3))
    tv3[:, :2] = tv
    tree.load_data(tv3, tf)  # *[16,2*np.pi,1],df)

    _, faces_markers = tree.look_up(d_v[sol_mark]*[16, 2*np.pi, 1])
    dblarea = igl.doublearea(tv, tf)
    if expand:
        face_markers = expand_color(tv, tf, faces_markers)
    col = -np.ones(len(tf))
    col[faces_markers] = 1/4*dblarea[faces_markers]
    return col


def amr_test_run(itemname, max_iterations = 5):
    gt_item = 'a0.01L2.npz.p2'

    # initial preparation
    with np.load('data/0925_tube/a0.01L2.npz') as npl:
        dense_v, dense_f = npl['ref_v'], npl['ref_f']
    dense_uv, dense_tf, dense_ptr = explode_uv_coord(dense_v, dense_f)

    saver = {}
    src_uv, src_tf, src_sol, _ = load_and_process('data/0925_tube/a0.01L2.npz.p2t2res.vtu')
    xf_sol = transfer_pymesh(src_uv, src_tf, src_sol,
                                 dense_uv, dense_tf)
    saver[gt_item] = xf_sol

    grad = igl.grad(dense_v[dense_ptr], dense_tf)

    for iteration in range(max_iterations):
        cmd = f"python pfplus.py tubes {itemname}.it{iteration}.npz --order=2 --steps=2 --suffix='p2t2' > {itemname}.it{iteration}.npz.log"
        print('>> Iterations ', iteration, 'Running >>', cmd)
        subprocess.run(cmd, shell=True)

        src_uv, src_tf, src_sol, src_newv = load_and_process(
            f'{itemname}.it{iteration}.npz.p2t2res.vtu')
        xf_sol = transfer_pymesh(src_uv, src_tf, src_sol,
                                 dense_uv, dense_tf)
        saver[f'it{iteration}.npz.p2'] = xf_sol
        with np.load(f'{itemname}.it{iteration}.npz', allow_pickle=True) as npl:
            uv0, tf0 = npl['uv']
        error = np.linalg.norm(
            (grad@(xf_sol - saver[gt_item])).reshape(3, -1, 3), axis=(0, 2))
        solution_marker = np.zeros(len(dense_uv), dtype=bool)
        solution_marker[np.unique(dense_tf[error > 0.5])] = True
        print(
            f'Percentage of Marker: {np.count_nonzero(solution_marker)}/{len(dense_uv)}')
        col = look_up(uv0, tf0, dense_uv, solution_marker,
                      expand=False)
        refine_v, refine_f = triangle_rolls.refine_tris(
            uv0, tf0, col, flag='-rYYqa')

        roll_v, roll_f = triangle_rolls.roll_up(refine_v, refine_f)
        roll_v /= 16
        tetv, tett = tube_gen.tetmesh_from_shell(
            roll_v*[1, .9, .9], roll_v, roll_f)
        print('min_vol', igl.volume(tetv, tett).min(), tett.shape)
        np.savez(f'{itemname}.it{iteration+1}.npz', uv=np.array((refine_v, refine_f), dtype=object),
                 ref_v=roll_v, ref_f=roll_f, tet_v=tetv, tet_t=tett)


if __name__ == '__main__':
    import fire
    fire.Fire(amr_test_run)
