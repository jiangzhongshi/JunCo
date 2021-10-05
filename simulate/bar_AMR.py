import triangle_rolls
import subprocess
import tube_gen
import numpy as np
import meshplot as mp
import meshio
import igl
import re
import glob
import sys
import meshzoo
sys.path.append('/home/zhongshi/Workspace/bichon/python/debug/')
import prism


def bar_gen(nx, ny):
    points, cells = meshzoo.cube(
        xmin=0.0, xmax=1.0,
        ymin=-0.05, ymax=0.05,
        zmin=-0.05, zmax=0.05,
        nx=nx, ny=ny, nz=ny
    )
    vol = igl.volume(points, cells)
    cells[vol < 0] = cells[vol < 0][:, [0, 1, 3, 2]]
    print('Diagnose', igl.volume(points, cells).min(), cells.shape)
    assert igl.volume(points, cells).min() > 0
    return points, cells


def load_tetgenio(filename):
    with open(filename + '.ele') as fp:
        tet = np.asarray([list(map(int, l.split()))
                         for l in fp.readlines()[1:] if '#' not in l])
        assert all(tet[:, 0] == np.arange(len(tet))), "Index assumed start with 0, use -z flag in TetGen"
        tet = tet[:, 1:]

    with open(filename + '.node') as fp:
        verts = np.asarray([list(map(float, l.split()))
                           for l in fp.readlines()[1:] if '#' not in l])
        assert all(verts[:, 0] == np.arange(len(verts))), "Index assumed start with 0, use -z flag in TetGen"
        verts = verts[:, 1:]
    return verts, tet


def write_tetgenio(path, vd, td, metrics):
    lines = [f'{len(vd)} 3 0 0\n']
    lines += [f'{i} ' + ' '.join(str(j)
                                 for j in v) + '\n' for i, v in enumerate(vd, 1)]
    elems = [f'{len(td)} 4 0\n']
    elems += [f'{i} ' + ' '.join(str(j)
                                 for j in t) + '\n' for i, t in enumerate(td+1, 1)]
    with open(path + '.node', 'w') as fp:
        fp.writelines(lines)
    with open(path + '.ele', 'w') as fp:
        fp.writelines(elems)
    if metrics is not None:
        met = [f'{len(vd)} 1\n'] + [f'{m}\n' for m in metrics]
        with open(path + '.mtr', 'w') as fp:
            fp.writelines(met)


def mesh_with_metric(dense_v, dense_t, dense_m):
    write_tetgenio('/tmp/tmptetgen/in.b', dense_v, dense_t, dense_m)

    tetgen_path = '/home/zhongshi/Workspace/tetgen/build/tetgen'
    info = subprocess.run(tetgen_path + ' -pqm /tmp/tmptetgen/in.off',
                          shell=True, capture_output=True)
    tv, tt = load_tetgenio('/tmp/tmptetgen/in.1')
    print('TetGen complete, output #v {}, #t {}'.format(len(tv), len(tt)))
    print(info.stdout.decode('utf-8'))
    return tv, tt

def tetgen_refine(in_v, in_t, marks):
    write_tetgenio('/tmp/tmptetgen/refine', in_v, in_t, None)
    vols = np.zeros(len(in_t))
    mark_vols = igl.volume(in_v, in_t[marks])
    vols[marks] = mark_vols/4

    vollines = [f'{len(in_t)}\n'] + [f'{i} {v}\n' for i,v in enumerate(vols,1)]

    with open('/tmp/tmptetgen/refine.vol','w') as fp: fp.writelines(vollines)
    info = subprocess.run('/home/zhongshi/Workspace/tetgen/build/tetgen -pqrazQ /tmp/tmptetgen/refine',
               shell=True,capture_output=True)
    tv, tt = load_tetgenio('/tmp/tmptetgen/refine.1')

    print('TetGen Refined, output #v {}, #t {}'.format(len(tv), len(tt)))
    return tv, tt

def load_vtu(name):
    m1 = meshio.read(name)
    v1, u1, t1 = m1.points, m1.point_data['solution'], m1.cells[0][1]
    _, uind, uinv = np.unique(np.round(v1 * 1e7).astype(int), return_index=True, return_inverse=True,
                              axis=0)
    new_v, new_sol, new_t = v1[uind], u1[uind], uinv[t1]
    return new_v, new_t, new_sol


def transfer_tetmesh(src_v0, src_t0, val0, tar_v1):
    tree = prism.AABB_tet(src_v0, src_t0)
    tid, bc1 = tree.point_bc(tar_v1)
    if len(val0.shape) == 1:
        val0 = val0.reshape(-1, 1)
    print('Failure Rate {}/{}'.format(np.count_nonzero(tid < 0), len(tid)))
    xf_x = np.einsum('ijk,ij->ik', val0[src_t0[tid]], bc1)
    # xf_x[tid==-1, :] = -1
    return xf_x, tid >= 0

import scipy.stats
def amr_run(itemname, max_iterations=5, dryrun=False):
    dense_v, dense_t, dense_sol = load_vtu(
        'data/1002_bar/bar100_8.npz.p2t2d0.1res.vtu')
    dense_metric = np.ones(len(dense_v))*0.1
    tet_v, tet_t = mesh_with_metric(dense_v, dense_t, dense_metric)
    
    threshold = 1e-3
    savers = [(dense_v, dense_t, dense_sol, np.arange(len(dense_v)))]
    for iteration in range(max_iterations):
        npzname = f'{itemname}.it{iteration}.npz'

        if not dryrun:
            np.savez(npzname, tet_v=tet_v,
                 tet_t=tet_t, metric=dense_metric)

        cmd = ' '.join(['python pfplus.py tubes', npzname,
        '--order=2 --steps=2 --suffix="p2t2d0.1" --tdelta=0.1',
        f'> {npzname}.log'])
        if dryrun:
            cmd = '# ' + cmd
        print('>> Iterations', iteration, 'Running >>', cmd)
        subprocess.run(cmd, shell=True)

        verts, tets, sols = load_vtu(f'{npzname}.p2t2d0.1res.vtu')
        xf_sol, valids = transfer_tetmesh(verts, tets, sols, dense_v)

        errors = np.linalg.norm((xf_sol - dense_sol)[valids], axis=1)
        error_pts = np.where(valids)[0][errors > threshold]
        print('>> Max Error', errors.max(), 'OT-Percentage', len(error_pts) / len(errors))

        if len(error_pts) == 0:
            return
        savers.append((tet_v, tet_t, xf_sol, valids))
        tree = prism.AABB_tet(tet_v, tet_t)
        tid, bc1 = tree.point_bc(dense_v[error_pts])
        tet_v, tet_t = tetgen_refine(tet_v, tet_t, np.unique(tid[tid >=0]))
    np.savez('temp.npz', savers)

if __name__ == '__main__':
    import fire
    fire.Fire(amr_run)
