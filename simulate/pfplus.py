import polyfempy as pf
import numpy as np
def setup_bc(v,t, sideset):
    solver = pf.Solver()
    solver.set_mesh(v,t)

    # x_max = v[:,0].max()
    # x_min = v[:,0].min()
    # x_mid = (x_max+x_min)/2
    # def sideset(p):
    #     if abs(p[0] - x_min) < 1e-3:
    #         return 2
    #     if abs(p[0] - x_max) < 1e-3:
    #         return 3
    #     return 1
    solver.set_boundary_side_set_from_bary(sideset)
    return solver

def setup_bc_pinch(v,t,radius=0.2):
    solver = pf.Solver()
    solver.set_mesh(v,t)

    x_max = v[:,0].max()
    x_min = v[:,0].min()
    x_mid = (x_max+x_min)/2
    def sideset(p):
        if abs(p[0] - x_min) < 1e-3 or abs(p[1] - x_min) < 1e-3:
            return 2
        if abs(p[0] - x_max) < 1e-3 or abs(p[1] - x_max) < 1e-3:
            return 2
        if np.linalg.norm(p) < radius:
            return 3
        return 1
    solver.set_boundary_side_set_from_bary(sideset)
    return solver

def solve_fem(solver, pde = pf.PDEs.LinearElasticity, 
              bc = dict(), materials = dict(E=2e2, nu=0.35), tol=1e-6):
    settings = pf.Settings(discr_order=2)
    problem = pf.Problem()

    settings.set_pde(pde)
    for t, v in materials.items():
        settings.set_material_params(t, v)

    for t, v in bc.items():
        if t.startswith('D'):
            problem.set_displacement(v[0], v[1])
        elif t.startswith('N'):
            problem.set_force(v[0],v[1])
        else:
            raise 1

    settings.set_problem(problem)

    solver.settings(settings)
    solver.solve()

    p, t, d = solver.get_sampled_solution()

    misises = solver.get_sampled_mises()

    pi_uni, indices, inverse = np.unique((p/tol).astype(int), 
                                         return_index=True, return_inverse=True, axis=0)
    t_uni = inverse[t]
    p_uni = p[indices]
    d_uni = d[indices]
    return p_uni, t_uni, d_uni, misises[indices]
