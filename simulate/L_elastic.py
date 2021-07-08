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

  
def check_stress(v,t,order):
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

    settings = pf.Settings(discr_order=order)
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

    pi_uni, indices, inverse = np.unique((p*1e8).astype(int), return_index=True, return_inverse=True, axis=0)
    p_uni = p[indices]
    t_uni = inverse[t]
    d_uni = d[indices]
    sys.stdout.flush()
    print(misises.max())
    return p_uni, t_uni, d_uni, misises[indices]

def load_tetgenio(filename):
    with open(filename + '.1.ele') as fp:
        tet = np.asarray([list(map(int,l.split())) for l in fp.readlines()[1:] if '#' not in l])
        assert all(tet[:,0] == np.arange(len(tet)))
        tet = tet[:,1:]

    with open(filename + '.1.node') as fp:
        verts = np.asarray([list(map(float,l.split())) for l in fp.readlines()[1:] if '#' not in l])
        assert all(verts[:,0] == np.arange(len(verts)))
        verts = verts[:,1:]
    return verts, tet

def main(mesh_type, cell_size, order):
    if mesh_type == 'L':
	    v, t = prepare_L_mesh(cell_size=cell_size)
    elif mesh_type == 'I':
        v, t = prepare_I_mesh(cell_size=cell_size)
    points ,tets, displ, stress = check_stress(v,t, order)

    np.savez(f'data/{mesh_type}stress-cell{cell_size}_p{order}.npz', p=points, t = tets, d=displ, s=stress)

if __name__ == '__main__':
	import fire
	fire.Fire(main)