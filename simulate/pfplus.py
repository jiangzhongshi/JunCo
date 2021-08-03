import numpy as np
import sys
import json
import tempfile
import subprocess

def main(input_file):
    t = 1.2
    bc_params = dict(
        dirichlet_boundary = [dict(id=1, value=[0.,0.,0.]),
                              dict(id=2, value=['-0.05',f'cos(t)*y + sin(t)*z - y',f'-sin(t)*y + cos(t)*z - z'])]
    )
    bnd_side = [
        dict(id=1, axis=-1, position=0.01),
        dict(id=2, axis=1, position=0.995),
    ]
    export_params = dict(vis_mesh=input_file+'.res', surface=True)
    d = dict(mesh=input_file,
             normalize_mesh=False,
             params=dict(E=2e4, nu=0.48),
             discr_order=2,
             problem='GenericTensor',
             tensor_formulation = "NeoHookean",
             problem_params = bc_params,
             export = export_params,
             boundary_sidesets=bnd_side,
             vismesh_rel_area = 0.1)
    print(json.dumps(d))
    sys.stdout.flush()
    with tempfile.NamedTemporaryFile(mode='r+', suffix='.json') as fp:
        fp.write(json.dumps(d))
        fp.seek(0)
        subprocess.run(['/home/zhongshi/Workspace/polyfem/build/PolyFEM_bin', '--cmd', '--json', fp.name], env = {'OMP_NUM_THREADS': '1'})

if __name__ == '__main__':
    import fire
    fire.Fire(main)

