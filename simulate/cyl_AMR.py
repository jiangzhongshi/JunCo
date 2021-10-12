from os import name
import triangle_rolls
import subprocess
import tube_gen
import numpy as np
import igl
import sys
sys.path.append('/home/zhongshi/Workspace/bichon/python/debug/')
import prism
import bar_AMR

import tube_gen
import tempfile
import logging

def closed_tube(n):
    v, f = tube_gen.tube(1, 1/16,n)

    leftie = np.where(v[:,0]==0)[0]
    srt_l = leftie[np.argsort(np.arctan2(v[leftie,1],v[leftie,2] ))]
    rightie = np.where(v[:,0]==1)[0]
    srt_r = rightie[np.argsort(np.arctan2(v[rightie,1],v[rightie,2] ))]

    v_l = np.array([0,0,0])
    v_r = np.array([1,0,0])

    f_l = [[len(v),b,a] for a,b in zip(srt_l, np.roll(srt_l,1))]
    f_r = [[len(v) + 1,a,b] for a,b in zip(srt_r, np.roll(srt_r,1))]


    closed_v = np.vstack([v, [v_l,v_r]])
    closed_f = np.vstack([f, f_l, f_r])
    
    tetgen_path = '/home/zhongshi/Workspace/tetgen/build/tetgen'
    with tempfile.TemporaryDirectory() as tmpdirname:
        igl.write_triangle_mesh(tmpdirname + '/in.off', closed_v, closed_f)
        info = subprocess.run(tetgen_path + f' -pqm {tmpdirname}/in.off',
                          shell=True, capture_output=True)
        tv, tt = bar_AMR.load_tetgenio(f'{tmpdirname}/in.1')
    return tv, tt

def refine_and_stretch(tet_v, tet_t, marker):
    def stretch(V, F, b, bc):
        slim = igl.SLIM(V, F, V,
                        b=b, bc=bc,
                        energy_type=igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET,
                        soft_penalty=1e5)
        slim.solve(10)
        return slim.vertices(), bc
    tet_v, tet_t = bar_AMR.tetgen_refine(tet_v, tet_t, marker)
    tet_f = igl.boundary_facets(tet_t)

    bc = igl.barycenter(tet_v, tet_f)

    bnd_pts = tet_f[0<bc[:,0]%1].flatten()

    side_freeze = list(set(tet_f.flatten()) - set(bnd_pts))

    target_pos = tet_v[bnd_pts].copy()
    target_pos[:,1:] = tet_v[bnd_pts,1:] / (np.linalg.norm(tet_v[bnd_pts,1:],axis=1,keepdims=True)*16)

    slim_v, _ = stretch(tet_v, tet_t, np.concatenate([bnd_pts, side_freeze]), 
                        bc = np.vstack([target_pos, tet_v[side_freeze]]))

    slim_v[tet_v[:,0]%1 == 0, 0] = tet_v[tet_v[:,0]%1 == 0, 0] # make sure the x-axis is strictly 0/1
    logging.debug(f'SLIM volumes {igl.volume(slim_v,tet_t).min()}')
    return slim_v, tet_t

import scipy.stats
def amr_run(itemname, max_iterations=5, dryrun=False):
    dense_v, dense_t, dense_sol = bar_AMR.load_vtu('data/1011_cyl/28k.npz.hangres.vtu', 1e-10)
    with np.load(itemname) as npl: tet_v, tet_t = npl['tet_v'], npl['tet_t']
    
    threshold = 3e-4
    savers = [(dense_v, dense_t, dense_sol, None)]
    suffix = 'hang'
    for iteration in range(max_iterations):
        npzname = f'{itemname}.it{iteration}.npz'

        if not dryrun:
            np.savez(npzname, tet_v=tet_v,
                 tet_t=tet_t)

        cmd = ' '.join(['python pfplus.py tubes', npzname,
        f'--order=2 --suffix="{suffix}" --setup="hang" --project_to_psd=False',
        f'> {npzname}.log'])
        if dryrun:
            cmd = '# ' + cmd
        logging.info(f'>> Iterations {iteration} Running >> {cmd}')
        subprocess.run(cmd, shell=True)

        verts, tets, sols = bar_AMR.load_vtu(f'{npzname}.{suffix}res.vtu')
        xf_sol, valids = bar_AMR.transfer_tetmesh(verts, tets, sols, dense_v)

        errors = np.linalg.norm((xf_sol - dense_sol)[valids], axis=1)
        error_pts = np.where(valids)[0][errors > threshold]
        logging.info(f'>> Max Error {errors.max()} OT-Percentage {len(error_pts) / len(errors)}')

        savers.append((tet_v, tet_t, xf_sol, valids, sols))
        if len(error_pts) == 0:
            break
        tree = prism.AABB_tet(tet_v, tet_t)
        tid, bc1 = tree.point_bc(dense_v[error_pts])
        tet_v, tet_t = refine_and_stretch(tet_v, tet_t, np.unique(tid[tid >=0]))
        # np.savez('temp.npz', tet_v = tet_v, tet_t = tet_t, marker = np.unique(tid[tid >=0]))
        # project to
        np.savez(f'{itemname}.{suffix}.saver.npz', savers)
    logging.info(f'length {len(savers)}')

if __name__ == '__main__':
    import fire
    # import coloredlogs
    from rich.logging import RichHandler

    logging.basicConfig( level="NOTSET", format="%(message)s", datefmt="[%m-%d %X]", handlers=[RichHandler(rich_tracebacks=True)])
    fire.Fire(amr_run)
