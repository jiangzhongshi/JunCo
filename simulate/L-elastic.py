import igl
import meshplot as mp
import numpy as np
import polyfempy as pf
import sys
def prepare_L_mesh(cell_size=0.1, scale=[1,1,1]):
    v,f = igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/cube.obj')
    v -= v.min(axis=0)
    v/=v.max()
    import pymesh
    um = pymesh.boolean(pymesh.form_mesh(v,f),
                   pymesh.form_mesh(v*[3,1,1]+np.array([1,0,0]), f),
                  operation='union')
    um = pymesh.boolean(um,
                   pymesh.form_mesh(v*[1,3,1]+np.array([0,1,0]), f),
                  operation='union')
    print('Start TetMesh')
    vm = pymesh.tetrahedralize(um, 
                               cell_size=cell_size,
                               engine ='tetgen')
    return vm.vertices, vm.voxels

def prepare_I_mesh(cell_size):
    import pymesh
    v,f = igl.read_triangle_mesh('/home/zhongshi/Workspace/libigl/tutorial/data/cube.obj')
    v -= v.min(axis=0)
    v/=v.max()
    import pymesh
    um = pymesh.boolean(pymesh.form_mesh(v,f),
                   pymesh.form_mesh(v*[3,1,1]+np.array([1,0,0]), f),
                  operation='union')
    print('Start TetMesh')
    vm = pymesh.tetrahedralize(um, 
                               cell_size=cell_size,
                               engine ='tetgen')
    return vm.vertices, vm.voxels

  
def check_stress(v,t):
    sys.stdout.flush()
    solver = pf.Solver()
    solver.set_mesh(v,t)
    x_max = v[:,0].max()
    x_min = v[:,0].min()
    def sideset(p):
        if p[0] > x_max - 1e-10:
            return 2
        if p[0] < x_min + 1e-10:
            return 3
        return 1
    solver.set_boundary_side_set_from_bary(sideset)

    settings = pf.Settings()
    problem = pf.Problem()

    settings.set_pde(pf.PDEs.LinearElasticity)

    settings.set_material_params("E", 200)
    settings.set_material_params("nu", 0.35)

    problem.set_displacement(3, [0, 0, 0])
    problem.set_force(2, [0, -1, 0])

    settings.set_problem(problem)

    solver.settings(settings)
    sys.stdout.flush()
    print()
    solver.solve()

    p, t, d = solver.get_sampled_solution()

    misises = solver.get_sampled_mises()

    p_uni, indices, inverse = np.unique(p, return_index=True, return_inverse=True, axis=0)
    t_uni = np.array([inverse[t[:, 0]], inverse[t[:, 1]], inverse[t[:, 2]], inverse[t[:, 3]]]).transpose()
    d_uni = d[indices, :]

    return p_uni, t_uni, d_uni, misises[indices]


def main(mesh_type, cell_size):
    if mesh_type == 'L':
	    v, t = prepare_L_mesh(cell_size=cell_size)
    elif mesh_type == 'I':
        v, t = prepare_I_mesh(cell_size=cell_size)
    points ,tets, displ, stress = check_stress(v,t)

    np.savez(f'data/{mesh_type}stress-cell{cell_size}.npz', p=points, t = tets, d=displ, s=stress)

if __name__ == '__main__':
	import fire
	fire.Fire(main)