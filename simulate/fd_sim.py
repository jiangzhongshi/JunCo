#!/usr/bin/env python
# coding: utf-8

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import firedrake as fd
from firedrake.cython import dmcommon
from datetime import datetime
from firedrake.cython import dmcommon
from firedrake.petsc import PETSc
from firedrake.utility_meshes import UnitCubeMesh, UnitSquareMesh
import meshio
import time
import sys
from scipy.spatial import ckdtree
import numpy as np

def construct_fd_mesh(V, C, bc_setter):
    plex = fd.mesh._from_cell_list(3, C, V, fd.COMM_WORLD)
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if (plex.getStratumSize("boundary_faces", 1) > 0):  # this is necessary for multi-core
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            idx = bc_setter(face_coords)
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, idx)
            # if face_coords[0] < 0.01 and face_coords[3] < 0.01 and face_coords[6] < 0.01:
                # plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            # elif face_coords[0] > 0.995 and face_coords[3] > 0.995 and face_coords[6] > 0.995:
                # plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            # else:
                # plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
    mesh = fd.Mesh(plex, reorder=None, distribution_parameters=None)
    return mesh

def mesh_to_VF(m):
    coordinates = m.coordinates
    coords = coordinates.dat.data_ro
    tri = coordinates.cell_node_map().values
    return coords, tri


from firedrake import *
def elastic_solve(mesh, bc_list, body_force = (0,0,0), settings = {}, materials = {}, parameters = {}):
    V = fd.VectorFunctionSpace(mesh, 'CG', 1)
    # Define functions
    du = TrialFunction(V)            # Incremental displacement
    v = TestFunction(V)             # Test function
    u = Function(V)                 # Displacement from previous iteration
    B = Constant(body_force)  # Body force per unit volume
    T = Constant((0.0,  0., 0.0))  # Traction force on the boundary

    # Kinematics
    Id = Identity(3)    # Identity tensor
    F = Id + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    E, nu = materials['E'], materials['nu']
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
    Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds
    F = derivative(Pi, u, v)
    J = derivative(F, u, du)

    x, y, z = SpatialCoordinate(mesh)
    dt = target_dt = settings['dt']
    t_end = settings['tmax']
    ti = 0
    while ti < t_end:
        if ti + dt > t_end:
            dt = t_end - ti
        ti += dt
        t = Constant(ti)
        print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]'), '>'*20, ti, dt)
        sys.stdout.flush()
        try:
            solve(F == 0, u, bc_list(V, x,y,z,t), J=J,
                solver_parameters=parameters)
        except fd.ConvergenceError:
            ti -= dt
            dt /= 2
            print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]'), 'dt bt', dt)
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

def hollow_ball(input_file):
    def bc_setter(face_coords):
        p = (face_coords).reshape(3,3).mean(axis=0)
        if np.linalg.norm(p - np.array([.5,.5,0])) < 0.1:
            return 1
        if np.linalg.norm(p - np.array([.5,.5,1])) < 0.1:
            return 2
        return 3
    def bc_list(V,x,y,z,t):
        c = fd.Constant([0.0, 0, 0.01*t])
        r = [fd.cos(t)*(x-0.5) + fd.sin(t) * (y-0.5)+ 0.5 - x, 
            -fd.sin(t)*(x-0.5) + fd.cos(t) * (y-0.5) + 0.5 - y, 0] 
        return [fd.DirichletBC(V, c, 1),
               fd.DirichletBC(V, r, 2)]

    start_time = time.time()
    mesh_read = meshio.read(input_file)
    mesh = construct_fd_mesh(mesh_read.points, mesh_read.cells[0][1], bc_setter)
    um,vm = elastic_solve(mesh, bc_list, 
            body_force = (0.0, 0.0, 0.0),
            settings = dict(dt=0.2, tmax=2),
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
    end_time = time.time()
    print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]'), f'Elapsed time: {(end_time - start_time)}')
    perm_id = find_perm(mesh.coordinates.dat.data_ro, 
                    mesh_read.points)
    disp = um.dat.data[perm_id]
    stress = vm.dat.data[perm_id]
    np.savez(input_file + '.fd.npz', disp = disp, stress = stress, cells = mesh_read.cells[0][1], verts = mesh_read.points)

    return mesh

def tubes(input_file):
    def bc_setter(face_coords):
        if face_coords[0] < 0.01 and face_coords[3] < 0.01 and face_coords[6] < 0.01:
            return 1
        if face_coords[0] > 0.995 and face_coords[3] > 0.995 and face_coords[6] > 0.995:
            return 2
        return 3

    def bc_list(V,x,y,z,t):
        c = fd.Constant([0.05*t, 0, 0])
        r = [fd.Constant(0), fd.cos(t)*y + fd.sin(t)*z - y,
                          -fd.sin(t)*y + fd.cos(t)*z - z]
        return [fd.DirichletBC(V, c, 1),
               fd.DirichletBC(V, r, 2)]

    mesh_read = meshio.read(input_file)
    mesh = construct_fd_mesh(mesh_read.points, mesh_read.cells[0][1], bc_setter)
    um,vm = elastic_solve(mesh, bc_list, 
            body_force = (0.0, 1e2, 0.0),
            settings = dict(dt=0.3, tmax=3),
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
    print(datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]'), 'Finished!! Save.')

# meshio.write('temp.msh', 
#              meshio.Mesh(points=mesh_read.points, 
#                          cells=[('tetra',mesh_read.cells[0][1])],
#                          cell_data={'gmsh:physical': [np.ones(540)], 
#                                   'gmsh:geometrical':[np.ones(540)] }),
#               file_format='gmsh22', binary=False)
# plex = Mesh('temp.msh')

if __name__ == '__main__':
    import gc
    gc.disable()

    import fire
    fire.Fire()