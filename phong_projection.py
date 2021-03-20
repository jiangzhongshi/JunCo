from vis_utils import h5deser, igl
import numpy as np
import numba
import tqdm
from scipy.optimize import least_squares
import pymesh


@numba.jit(nopython=True)
def phong_coeffs(V0, V1, V2, N0, N1, N2, Q):
    return (np.cross(V1-V0, N1-N0),  # a2
            np.cross(V2-V0, N2-N0),  # b2
            np.cross(V1-V0, N2-N0)+np.cross(V2-V0, N1-N0),  # ab
            np.cross(V0-Q, N1-N0)+np.cross(V1-V0, N0),  # a
            np.cross(V2-V0, N0) + np.cross(V0-Q, N2-N0),  # b
            np.cross(V0-Q, N0))


@numba.jit(nopython=True)
def quadr(x, coef):
    res = np.zeros(3)
    for k, (i, j) in enumerate([(2, 0), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)]):
        res += coef[k]*x[0]**i*x[1]**j
    return res


@numba.jit(nopython=True)
def jac(x, coef):
    res = np.zeros((3, 2))
    for k, (i, j) in enumerate([(2, 0), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)]):
        res[:, 0] += coef[k]*(i)*x[0]**(i-1)*x[1]**j
        res[:, 1] += coef[k]*(j)*x[0]**(i)*x[1]**(j-1)
    return res

def search_in_tri(v, f0, V1,F1,VN1, TT, visited):
    if visited[f0] == 1:
        return None
    f = F1[f0]
    coef = phong_coeffs(*V1[f], *VN1[f], v)
    sol = least_squares(quadr, x0=np.array([1/3,1/3]),jac=jac, bounds=(0,1),
                        args=([coef]))
    x = sol.x
    visited[f0] = 1
    if sol.cost <= 1e-8 and x.sum() <= 1:
        return (f0,x[0],x[1])
    else:
        for j in TT[f0]:
            y = search_in_tri(v,j,V1,F1,VN1,TT,visited)
            if y is not None:
                return y
def main():
    refV, refF, _, _, _, sF, _ = h5deser('../buildr/scalability/shark.obj.h5')
    _, V1, F1, _, _ = igl.qslim(refV, refF, len(sF))
    VN1 = igl.per_vertex_normals(V1, F1)

    queryV = -np.ones_like(refV)

    _, closest_faces, _ = pymesh.distance_to_mesh(pymesh.form_mesh(V1,F1),refV)
    TT, _ = igl.triangle_triangle_adjacency(F1)

    for vid, (v,f0) in enumerate(zip(tqdm.tqdm(refV), closest_faces)):
        visited = np.zeros(len(F1))
        x = search_in_tri(v, f0, V1,F1,VN1, TT, visited)
        if x is None:
            continue
        else:
            queryV[vid] = x
    np.save("queryV.npy",queryV)

if __name__ == '__main__':
    main()
