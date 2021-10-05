import numpy as np
import sys
import json
import tempfile
import subprocess
import os
import meshio
import ctypes
import shutil


def pf_run(d, input_file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        if input_file.endswith('npz'):
            with np.load(input_file) as npl:
                tet_v, tet_t = npl['tet_v'], npl['tet_t']
                meshio.write(tmpdirname + '/input.msh', meshio.Mesh(points=tet_v,
                             cells=[('tetra', tet_t.astype(ctypes.c_size_t))]))
                d['mesh'] = tmpdirname + '/input.msh'
        print(json.dumps(d))
        sys.stdout.flush()
        with open(tmpdirname + '/cmd.json', 'w') as fp:
            fp.write(json.dumps(d))

        pf_bin = '/home/zhongshi/Workspace/polyfem/build/PolyFEM_bin'
        if d['check_hess']:
            pf_bin = '/home/zhongshi/Workspace/polyfem/build_hess/PolyFEM_bin'
            # pf_bin = '/home/zhongshi/Workspace/polyfem/build/PolyFEM_hess_bin'

        num_threads = d['num_threads']
        suffix = d['suffix']
        info = subprocess.run([pf_bin,
                               '--log_level', '1',
                               '--max_threads', str(num_threads),
                               '--cmd',
                               '-o', tmpdirname,
                               '--json', tmpdirname + '/cmd.json'],
                              env=dict(OMP_NUM_THREADS=str(num_threads)))
        if info.returncode == 0:
            shutil.move(f'{tmpdirname}/vis.vtu', '{input_file}.{suffix}res.vtu')
        else:
            print('WARNING: PolyFEM failed')


def bc_zoo(key):
    zoo = dict()
    zoo['twist'] = dict(problem_params=dict(
        dirichlet_boundary=[dict(id=1, value=['0.05*t', '0', '0']),
                            dict(id=2, value=['0',
                                              f'cos(t)*y + sin(t)*z - y',
                                              f'-sin(t)*y + cos(t)*z - z'])],
        is_time_dependent=True,
        rhs=[0, 10, 0],
    ),
        boundary_sidesets=[
        dict(id=1, axis=-1, position=0),
        dict(id=2, axis=1, position=1),
    ])
    zoo['hang'] = dict(problem_params=dict(
        dirichlet_boundary=[dict(id=1, value=['0.0', '0', '0']),
                            dict(id=2, value=['0', '0', '0'])],
        is_time_dependent=False,
        rhs=[0, 10, 0]
    ),
        boundary_sidesets=[
        dict(id=1, axis=-1, position=0.01),
        dict(id=2, axis=1, position=0.995),
    ])
    zoo['hollowball'] = dict(problem_params = dict(
        dirichlet_boundary=[dict(id=1, value=[0., 0., 0.], dimension=[True, True, False]),
                            dict(id=2, value=[f'cos(t)*(x-0.5) + sin(t) * (y-0.5)+ 0.5 - x',
                                              f'-sin(t)*(x-0.5) + cos(t) * (y-0.5) + 0.5 - y', '0'])],
        is_time_dependent=True,
        rhs=[0, 0, -0.5],
    ),
    boundary_sidesets = [
        dict(id=1, center=[.5, .5, 0], radius=0.1),
        dict(id=2, center=[.5, .5, 1], radius=0.1),
    ])
    return zoo[key]


def tubes(input_file, order=1, n_refs=0, steps=10, tdelta=0.3, suffix='', setup='twist', **kwargs):
    d = dict(mesh=input_file,
             normalize_mesh=False,
             tend=tdelta*steps,
             time_steps=steps,
             params=dict(E=2e4, nu=0.48, rho=1e2),
             problem='GenericTensor',
             tensor_formulation="NeoHookean",
             discr_order=order,
             n_refs=n_refs,
             project_to_psd=True,
             solver_params=dict(nl_iterations=500),
             export=dict(vis_mesh='vis.vtu'),
             #  problem_params = prob_params,
             #  boundary_sidesets=bnd_side,
             vismesh_rel_area=1e-5)
    d.update(bc_zoo(setup))
    if d['discr_order'] > 1:
        d['lump_mass_matrix'] = True
    d.update(dict(suffix=suffix, num_threads=2, check_hess=False))
    d.update(kwargs)
    
    pf_run(d, input_file)

if __name__ == '__main__':
    import fire
    fire.Fire()
