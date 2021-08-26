#!/usr/bin/env python
# coding: utf-8


import firedrake as fd
from firedrake.cython import dmcommon
import datetime
from firedrake.cython import dmcommon
from firedrake.petsc import PETSc
from firedrake.utility_meshes import UnitCubeMesh, UnitSquareMesh
import meshio
import time
import sys
from scipy.spatial import ckdtree

def construct_fd_mesh(V, C):
    plex = fd.mesh._from_cell_list(3, C, V, fd.COMM_WORLD)
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if (plex.getStratumSize("boundary_faces", 1) > 0):  # this is necessary for multi-core
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if face_coords[0] < 0.01 and face_coords[3] < 0.01 and face_coords[6] < 0.01:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            elif face_coords[0] > 0.995 and face_coords[3] > 0.995 and face_coords[6] > 0.995:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
    mesh = fd.Mesh(plex, reorder=None, distribution_parameters=None)
    return mesh

def mesh_to_VF(m):
    coordinates = m.coordinates
    coords = coordinates.dat.data_ro
    tri = coordinates.cell_node_map().values
    return coords, tri


from firedrake import *
def run_solve(mesh, V, materials, parameters):


    # Define functions
    du = TrialFunction(V)            # Incremental displacement
    v = TestFunction(V)             # Test function
    u = Function(V)                 # Displacement from previous iteration
    B = Constant((0.0, 1e2, 0.0))  # Body force per unit volume
    T = Constant((0.0,  0., 0.0))  # Traction force on the boundary

    # Kinematics
    Id = Identity(3)    # Identity tensor
    F = Id + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    # Elasticity parameters
    E, nu = materials['E'], materials['nu']#10.0, 0.3
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)

    x, y, z = SpatialCoordinate(mesh)
    ti = 0
    dt = target_dt = 0.3
    while ti < 3:
        ti += dt
        t = Constant(ti)
        print(ti, dt)
        c = Constant([0.05*t, 0, 0])
        r = [Constant(0), fd.cos(t)*y + fd.sin(t)*z - y,
                          -fd.sin(t)*y + fd.cos(t)*z - z]
        bcs = [DirichletBC(V, c, 1),
               DirichletBC(V, r, 2)]
        try:
            solve(F == 0, u, bcs, J=J,
                solver_parameters=parameters)
        except fd.ConvergenceError:
            ti -= dt
            dt /= 2
            print('dt bt', dt)
        else:
            dt = (target_dt + dt)/2
        
    def sigma(u):
        return lmbda*fd.div(u)*Id + mu*(fd.grad(u) + fd.grad(u).T)
    def von_mises(u):
        s = sigma(u) - (1./3)*fd.tr(sigma(u)) *             fd.Identity(3)  # deviatoric stress
        vonmises = fd.sqrt(3./2*fd.inner(s, s))
        return vonmises

    vm = von_mises(u)
    FS1 = FunctionSpace(mesh, 'CG', 1)
    vm = fd.project(vm, FS1)
    return u, vm



def find_perm(v0,v1):
    dist, idx = ckdtree.cKDTree(v0).query(v1)
    return idx


def tubes(input_file):
    mesh_read = meshio.read(input_file)
    mesh = construct_fd_mesh(mesh_read.points, mesh_read.cells[0][1])
    V = fd.VectorFunctionSpace(mesh, 'CG', 1)
    um,vm = run_solve(mesh, V, 
              materials = dict(E=2e4, nu=0.48, rho=2e3),
              parameters= {
            'snes_type': 'newtonls',
            'snes_linesearch_type':'bt',
            'snes_linesearch_monitor': None,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'ksp_rtol': 1e-10,
            'ksp_atol': 1e-10,
            'snes_monitor': None,
        })


    perm_id = find_perm(mesh.coordinates.dat.data_ro, 
                    mesh_read.points)

    disp = um.dat.data[perm_id]
    stress = vm.dat.data[perm_id]
    np.savez(input_file + '.fd.npz', disp = disp, stress = stress, cells = mesh_read.cells[0][1], verts = mesh_read.points)

# meshio.write('temp.msh', 
#              meshio.Mesh(points=mesh_read.points, 
#                          cells=[('tetra',mesh_read.cells[0][1])],
#                          cell_data={'gmsh:physical': [np.ones(540)], 
#                                   'gmsh:geometrical':[np.ones(540)] }),
#               file_format='gmsh22', binary=False)
# plex = Mesh('temp.msh')

if __name__ == '__main__':
    import fire
    fire.Fire(tubes)