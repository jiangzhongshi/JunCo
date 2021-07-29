"""
Pipeline file for simplified simulation.
"""

import subprocess
import fire
import meshio
import pfplus
import L_elastic
import sys
sys.path.append('..')
from meshplotplus.plot import scale
from vis_utils import h5reader
import polyfempy as pf
import meshplot as mp
import numpy as np
import igl
import os
import pymesh

def extract_tetra_surface(filename):
    """
    Read in tetwild mesh, then return the path of the given velocity and time
    """
    # Read in the tetwild mesh
    v, t = igl.read_msh(filename)
    v1, f1, _, _ = igl.remove_unreferenced(v, igl.boundary_facets(t))
    V1 = scale(v1)
    # save msh, and surface for shell computation.
    return v1, f1, len(t)


def read_shell_file(filename):
    """
    Read in the shell file.
    """
    mV, mF = h5reader(filename, 'mV', 'mF')
    m = pymesh.tetrahedralize(pymesh.form_mesh(mV, mF), cell_size=1.0)
    v, t = m.vertices, m.voxels

    # in fact, it does not matter to scale surface only or all verts. Since min and max are taken.
    # so in this case, scale with mean is bad idea.
    return scale(v), t


def meshgen(filename, force_rerun=False):
    tet_folder = '/home/zhongshi/data/ftetwild_output_msh/'
    surf_folder = '/home/zhongshi/data/simpsim/surf/'
    shell_folder = '/home/zhongshi/data/simpsim/shell/'
    tetout_folder = '/home/zhongshi/data/simpsim/tetout/'
    v1, f1, tnum = extract_tetra_surface(tet_folder + filename)
    trimesh = surf_folder + filename + '.ply'
    igl.write_triangle_mesh(trimesh, v1, f1)  # binary

    shellpath = shell_folder + filename + '.ply.h5'
    if not os.path.exists(shellpath):
        force_rerun = True
    if force_rerun:
        cmd = f'./cumin_bin -i {trimesh} -o {shell_folder} -l {shell_folder}/log --control-skip_volume --control-no-enable_curve --feature-dihedral_threshold 0.6'
        print(cmd)
        ret = subprocess.run(args=cmd.split(), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd='/home/zhongshi/Workspace/bichon/buildr/')
        print('Returning', ret)
    
    v2, t2 = read_shell_file(shellpath)
    m = meshio.Mesh(points=v2, cells = [('tetra',t2)])
    meshio.write(tetout_folder + filename + '.msh', m, binary=True)
    print(f'{tnum} -> {t2.shape[0]}')

import polyfempy as pf
import meshplot as mp
import numpy as np
import igl
import sys

def align_mainaxis(v):
    align_mainaxis.a = v.mean(axis=0)
    v1 = v-align_mainaxis.a

    align_mainaxis.b = v1.max()
    v1 /= align_mainaxis.b

    _ , dirs = np.linalg.eig(v1.T@v1)
    if (np.linalg.det(dirs) < 0):
        dirs = dirs[:,[0,2,1]]
    align_mainaxis.c = dirs
    return v1@(dirs)
def simulate_stretch(v,t):
    x_max = v[:,0].max()
    x_min = v[:,0].min()
    x_mid = (x_max+x_min)/2
    length = 0.2
    def sideset(p):
        if abs(p[0] - x_min) < length:
            return 2
        if abs(p[0] - x_max) < length:
            return 3
        return 1

    pt,tet,disp,vmises = pfplus.solve_fem(pfplus.setup_bc(v,t,sideset),
          pde= pf.PDEs.NonLinearElasticity,
          bc = dict(D1=(3, [0.2,0,0]),
                    D0=(2, [0,0,0])),
          materials = dict(E=2e2, nu=0.35))
    return pt, tet, disp, vmises

def compare_sim(filename):
    tet_folder = '/home/zhongshi/data/ftetwild_output_msh/'
    tetout_folder = '/home/zhongshi/data/simpsim/tetout/'
    sim_folder = '/home/zhongshi/data/simpsim/simres/'
    m = meshio.read(tet_folder + filename)
    v, t = m.points, m.cells[0][1]
    v = scale(v)
    v = align_mainaxis(v)

    m1 = meshio.read(tetout_folder + filename + '.msh')
    v1, t1 = m1.points, m1.cells[0][1]
    v1 = scale(v1)
    v1 = ((v1 - align_mainaxis.a)/align_mainaxis.b)@align_mainaxis.c

    info0 = simulate_stretch(v,t)
    print('='*10 + 'simulate next')
    info1 = simulate_stretch(v1,t1)
    np.savez(sim_folder + filename, tw=info0, tg=info1)


if __name__ == '__main__':
    fire.Fire()
