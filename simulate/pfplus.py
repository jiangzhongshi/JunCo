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

        num_threads = d['num_threads']
        suffix = d['suffix']
        info = subprocess.run([pf_bin,
                               '--log_level', '1',
                               '--max_threads', str(num_threads),
                               '--cmd',
                               '-o', tmpdirname,
                               '--json', tmpdirname + '/cmd.json'],
                              env=dict(OMP_NUM_THREADS=str(num_threads)))
        if info.returncode == 0 and suffix != 'nosave':
            shutil.move(f'{tmpdirname}/vis.vtu', f'{input_file}.{suffix}res.vtu')
            print('Success')
        else:
            print('WARNING: PolyFEM failed')


def bc_zoo(key):
    with open('bc_zoo.json') as fp:
        zoo = json.load(fp)
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
             project_to_psd=False,
             solver_params=dict(nl_iterations=500),
             export=dict(vis_mesh='vis.vtu'),
             #  problem_params = prob_params,
             #  boundary_sidesets=bnd_side,
             vismesh_rel_area=1e-5)
    d.update(bc_zoo(setup))
    if d['discr_order'] > 1:
        d['lump_mass_matrix'] = True
    d.update(dict(suffix=suffix, num_threads=2, check_hess=False))
    merge_args(d, kwargs)

    pf_run(d, input_file)

def merge_args(m_d, kwargs):
    def dict_merge(dct, merge_dct):
        for k, v in merge_dct.items():
            if (k in dct and isinstance(v, dict)):
                dict_merge(dct[k], v)
            else:
                dct[k] = v
    def key2dict(k,v):
        newdict = dict()
        if '.' in k:
            first, rest = k.split('.',1)
            newdict[first] = key2dict(rest, v)
        else:
          newdict[k] = v
        return newdict

    for k, v in kwargs.items():
      dict_merge(m_d, key2dict(k,v))


if __name__ == '__main__':
    import fire
    fire.Fire()
