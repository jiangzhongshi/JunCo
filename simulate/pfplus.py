import numpy as np
import sys
import json
import tempfile
import subprocess

def main(input_file):
    prob_params = dict(
        dirichlet_boundary = [dict(id=1, value=['0','1e-2*t','1e-2*t']),
                              dict(id=2, value=['0',
                              f'cos(t)*y + sin(t)*z - y',
                              f'-sin(t)*y + cos(t)*z - z'])],
        is_time_dependent = True,
        rhs = [0,0.81,0],
  )
    bnd_side = [
        dict(id=1, axis=-1, position=0.01),
        dict(id=2, axis=1, position=0.995),
    ]
    export_params = dict(vis_mesh=input_file+'.res', surface=True)
    d = dict(mesh=input_file,
            tend = 2.0,
             normalize_mesh=False,
             params=dict(E=2e4, nu=0.48, rho=2e3),
             discr_order=2,
             problem='GenericTensor',
             tensor_formulation = "NeoHookean",
             problem_params = prob_params,
             line_search='bisection',
             solver_params = dict(nl_iterations=500),
             export = export_params,
             boundary_sidesets=bnd_side,
             vismesh_rel_area = 0.1)
    print(json.dumps(d))
    sys.stdout.flush()
    with tempfile.NamedTemporaryFile(mode='r+', suffix='.json') as fp:
        fp.write(json.dumps(d))
        fp.seek(0)
        subprocess.run(['/home/zhongshi/Workspace/polyfem/build/PolyFEM_bin', '--log_level', '1',  '--cmd', '--json', fp.name, '--output_dir', 'data/'],  # tubes
                        env = dict(OMP_NUM_THREADS='4'))

def hollow_ball(input_file):
    #  = 'simulate/data/math_form_1_obj.msh'
    t = 't' # time dependent variable
    bc_params = dict(
        dirichlet_boundary = [dict(id=1, value=[0.,0.,0.]),
        dict(id=2, value = [f'cos({t})*(x-0.5) + sin({t}) * (y-0.5)+ 0.5 - x', 
                            f'-sin({t})*(x-0.5) + cos({t}) * (y-0.5) + 0.5 - y', '-0.01*t'])], 
        is_time_dependent = True,
    )
    bnd_side = [
        dict(id=1, center=[.5,.5,0], radius=0.1),
        dict(id=2, center=[.5,.5,1], radius=0.1),
    ]
    export_params = dict(vis_mesh=input_file+'.res', surface=True)
    d = dict(mesh=input_file,
            normalize_mesh=False,
            params=dict(E=2e4, nu=0.48, density=2e3),
            # time_steps = 100,
            tend = 2.0,
            discr_order=2,
            problem='GenericTensor',
            tensor_formulation = "NeoHookean",
            # tensor_formulation = "LinearElasticity",
            problem_params = bc_params,
            line_search='bisection',
            solver_params = dict(nl_iterations=300),
            export = export_params,
            boundary_sidesets=bnd_side,
            vismesh_rel_area = 0.1)

    import tempfile
    import json
    import subprocess
    with tempfile.NamedTemporaryFile(mode='r+', suffix='.json') as fp:
            fp.write(json.dumps(d))
            fp.seek(0)
            subprocess.run(['/home/zhongshi/Workspace/polyfem/build/PolyFEM_bin', '--log_level', '0', '--cmd', '--json', fp.name], 
                        env = dict(OMP_NUM_THREADS='4'))
if __name__ == '__main__':
    import fire
    fire.Fire()

