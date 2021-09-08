import numpy as np
import sys
import json
import tempfile
import subprocess
import os
import fire

def pf_run(d):
    print(json.dumps(d))
    sys.stdout.flush()
    with tempfile.NamedTemporaryFile(mode='r+', suffix='.json') as fp:
      fp.write(json.dumps(d))
      fp.seek(0)
      subprocess.run(['/home/zhongshi/Workspace/polyfem/build/PolyFEM_bin', '--log_level', '1',  '--cmd', '--json', fp.name, '-o', 'data/'],  # tubes
                          env = dict(OMP_NUM_THREADS='1'))
def box(input_file, suffix):
    prob_params = dict(
      dirichlet_boundary=[dict(id=1, value=[0, 0, 0])],
                # dict(id=2, value=[0, -0.5, 0])],
      neumann_boundary=[dict(id=2, value=[0, 1, 0])]
    )
    bnd_side = [
          dict(id=1, axis=-2, position=0.01),
          dict(id=2, axis=2, position=0.995),
    ]
    export_params = dict(vis_mesh=os.path.basename(input_file) + suffix + '.res.vtu')#, surface=True)
    d = dict(mesh=input_file,
              normalize_mesh=False,
              params=dict(E=1, nu=0.3),
              discr_order=2,
              problem='GenericTensor',
              tensor_formulation = "LinearElasticity",
              # tensor_formulation = "NeoHookean",
              problem_params = prob_params,
              has_collision = False,
              project_to_psd=True,
              solver_params = dict(nl_iterations=500),
              export = export_params,
              boundary_sidesets=bnd_side,
              vismesh_rel_area = 0.1)
    pf_run(d)

def twobox(input_file, suffix):
    multimesh = [
      dict(mesh = input_file, position = [0,0.00], scale = 0.5, body_id = 1), 
      dict(mesh = input_file, position = [0,0.51], scale = 0.5, body_id = 2)
    ]
  
    bnd_side = [
          dict(id=1, axis=-2, position=0.01),
          dict(id=2, axis=2, position=0.995),
          dict(id=3, center=[0,0.75], radius = 0.1),
    ]

    boundary_conditions = [('d', dict(id=1, value=[0, 0])),
                           ('d', dict(id=2, value=[0, 0])),
                            ('n', dict(id=3, value=[-0.1, -0.25]))]

    export_params = dict(vis_mesh=os.path.basename(input_file) + suffix + '.res.vtu')

    prob_params = dict(dirichlet_boundary = [], neumann_boundary =[])
    for t, bc in boundary_conditions:
      if t == 'd':
        prob_params['dirichlet_boundary'].append(bc)
      elif t == 'n':
        prob_params['neumann_boundary'].append(bc)
      else:
        raise RuntimeError("unknown bc type!")
    d = dict(meshes = multimesh,
              normalize_mesh=False,
              params=dict(E=1, nu=0.3),
              discr_order=2,
              problem='GenericTensor',
              tensor_formulation = "LinearElasticity",
              # tensor_formulation = "NeoHookean",
              problem_params = prob_params,
              has_collision = True,
              project_to_psd=True,
              solver_params = dict(nl_iterations=500),
              export = export_params,
              boundary_sidesets=bnd_side,
              vismesh_rel_area = 0.1)
    pf_run(d)

if __name__ == '__main__':
  fire.Fire()
