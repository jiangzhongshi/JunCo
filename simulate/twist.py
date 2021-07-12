verts, tets = L_elastic.load_tetgenio('temp/bar')

verts, tets = L_elastic.prepare_I_mesh(10)
print(tets.shape)

import polyfempy as pf
v,t = verts, tets

solver = pf.Solver()
solver.set_mesh(v,t)

x_max = v[:,0].max()
x_min = v[:,0].min()
x_mid = (x_max+x_min)/2
def sideset(p):
    if abs(p[0] - x_min) < 1e-3:
        return 2
    if abs(p[0] - x_max) < 1e-3:
        return 3
    return 1
solver.set_boundary_side_set_from_bary(sideset)

settings = pf.Settings(discr_order=2)
problem = pf.Problem()

settings.set_pde(pf.PDEs.NonLinearElasticity)

settings.set_material_params("E", 2e4)
settings.set_material_params("nu", 0.35)

problem.set_displacement(2, [0, 0, 0])
theta = np.pi
problem.set_displacement(3, ['0',f'cos({theta})*(y-0.5) - sin({theta})*(z-0.5) + 0.5 - y',f'sin({theta})*(y-0.5)+cos({theta})*(z-0.5) + 0.5 - z'])

settings.set_problem(problem)

solver.settings(settings)
solver.solve()

p, t, d = solver.get_sampled_solution()

misises = solver.get_sampled_mises()

pi_uni, indices, inverse = np.unique((p*1e3).astype(int), 
                                     return_index=True, return_inverse=True, axis=0)
t_uni = inverse[t]
p_uni = p[indices]
d_uni = d[indices]
misises[indices].max()