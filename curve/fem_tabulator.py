#!/usr/bin/env python
'''File is original from sympy repo, with modifications, fixes etc'''

import collections
import itertools
import tqdm
import numba
import sympy
from sympy import Rational, zeros, diff
import math
import functools
import h5py
import numpy as np

def codecs():
    def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i+1]*j]

    # print(np.array2string(tetra35_codec_n,separator=',').replace('[','{').replace(']','}') )
    def decompose_x(x): return np.array([list(map(int, c))
                                         for c in x.split(',')])
    # Quadratic Codec
    tri6 = '00,11,22,01,12,02'
    tri6_n = decompose_x(tri6)
    tri6_op = np.eye(3)[tri6_n].mean(axis=1)

    # Cubic Codecs
    tetra20_codec = '000,111,222,333,001,011,112,122,022,002,033,003,233,223,133,113,012,013,023,123'
    tri10_codec = '000,111,222,001,011,112,122,022,002,012'

    tetra20_codec_n = np.array([list(map(int, c))
                                for c in tetra20_codec.split(',')])
    tetra20_op = np.eye(4)[tetra20_codec_n].mean(axis=1)
    tri10_codec_n = np.array([list(map(int, c))
                              for c in tri10_codec.split(',')])
    tri10_op = np.eye(3)[tri10_codec_n].mean(axis=1)

    # Quartic Codecs
    tetra35_codec = '0000,1111,2222,3333,0001,0011,0111,1112,1122,1222,0222,0022,0002,0333,0033,0003,2333,2233,2223,1333,1133,1113,0012,0122,0112,0013,0113,0133,0023,0233,0223,1233,1123,1223,0123'
    tetra35_codec_n = np.array([list(map(int, c))
                                for c in tetra35_codec.split(',')])
    tetra35_op = np.eye(4)[tetra35_codec_n].mean(axis=1)
    tri15_codec = '0000,1111,2222,0001,0011,0111,1112,1122,1222,0222,0022,0002,0012,0122,0112'
    tri15_codec_n = np.array([list(map(int, c))
                              for c in tri15_codec.split(',')])
    tri15_op = np.eye(3)[tri15_codec_n].mean(axis=1)

    tetra35_vtk = '0000,1111,2222,3333,0001,0011,0111,1112,1122,1222,0222,0022,0002,0003,0033,0333,1113,1133,1333,2223,2233,2333,0013,0113,0133,1223,1233,1123,0023,0233,0223,0012,0122,0112,0123'
    wedge18_codec = '''0000,1111,2222,3333,4444,5555,0011,0022,0033,1122,1144,2255,3344,3355,4455,0134,0235,1245'''
    p1wedge = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 1.],
                        [0., 1., 1.]])
    wedge18_codec_n = np.array([list(map(int, w))
                                for i, w in enumerate(wedge18_codec.split(','))])
    wedge_elev_op = np.eye(6)[wedge18_codec_n].mean(axis=1) @ p1wedge
    return dict(tri6=[tri6, tri6_n, tri6_op],
                tetra20=[tetra20_codec, tetra20_codec_n, tetra20_op],
                tri10=[tri10_codec, tri10_codec_n, tri10_op],
                tetra35=[tetra35_codec, tetra35_codec_n, tetra35_op],
                tri15=[tri15_codec, tri15_codec_n, tri15_op])


''' temporay note: gmsh wedge 18 ordering
0000,1111,2222,3333,4444,5555,0011,0022,0033,1122,1144,2255,3344,3355,4455,0134,0235,1245
(1/2)*array([[0, 0, 0],
       [2, 0, 0],
       [0, 2, 0],
       [0, 0, 2],
       [2, 0, 2],
       [0, 2, 2],
       [1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [1, 1, 0],
       [2, 0, 1],
       [0, 2, 1],
       [1, 0, 2],
       [0, 1, 2],
       [1, 1, 2],
       [1, 0, 1],
       [0, 1, 1],
       [1, 1, 1]])
'''


def tuple_gen(order, var_n):
    if var_n == 0:
        return [[order]]
    l = []
    for i in range(order + 1):
        r = tuple_gen(order-i, var_n - 1)
        l += [[i]+t for t in r]
    l = sorted(l, key=lambda x: (-sum(i**2 for i in x), x[::-1]))
    return l


def bernstein_space(order, nsd):
    expr = 0
    basis = []
    coeff = []

    mc = sympy.multinomial_coefficients(nsd + 1, order)
    var = [sympy.Symbol(f'x{i}') for i in range(nsd)]
    var = [1-sum(var)] + var
    for tup, fac in mc.items():
        aij = sympy.Symbol('a_{}'.format('_'.join(map(str, tup))))
        term = functools.reduce(
            sympy.Mul, (v**e for v, e in zip(var, tup)))
        expr += aij*fac*term
        basis.append(fac*term)
        coeff.append(aij)

    return expr, coeff, basis


def create_point_set(codec):
    dim = len(codec[1]) - 1
    codec = np.asarray(codec)
    h = Rational(1, codec[0].sum())
    corners = sympy.Matrix(np.vstack([np.zeros(dim), np.eye(dim)]).astype(int))
    return [sympy.Matrix(r).T*corners*h for r in codec]


def create_matrix(equations, coeffs):
    '''Extract the coefficients to a matrix
    A is used for evaluate at Lagrange nodes. Therefore, A(Bnodes) = Lnodes.
    '''
    A = zeros(len(equations))
    for j in range(len(coeffs)):
        c = coeffs[j]
        for i in range(len(equations)):
            A[i, j] = equations[i].coeff(c)
    return A


def bernstein_evaluator(x, y, z, codecs):
    m = len(codecs[0]) # dim + 1
    n = codecs[0][0] # order
    mc_dict = sympy.multinomial_coefficients(m, n)
    mc = np.array([mc_dict[tuple(c)] for c in codecs])

    w = 1-x-y-z
    computed_powers = np.array([(w**i, x**i, y**i, z**i)
                              for i in range(n + 1)])  # make use of 0**0 == 1
    return mc[:,None]*np.array(
        [np.prod([computed_powers[c, i] for i, c in enumerate(cod)], axis=0) for cod in codecs])

def bernstein_deri_evaluator(x, y, z, codecs):
    m = len(codecs[0]) # dim + 1
    n = codecs[0][0] # order
    mc_dict = sympy.multinomial_coefficients(m, n)
    mc = np.array([mc_dict[tuple(c)] for c in codecs])

    w = 1-x-y-z
    computed_powers = np.array([(w**i, x**i, y**i, z**i)
                              for i in range(n + 1)])  # make use of 0**0 == 1
    dx_dy_dz = np.zeros((3, len(codecs))+x.shape)  # 3,35,data(8,220)
    for d in range(3):
        for ci, (mn, cod) in enumerate(zip(mc, codecs)):
            p = cod[d+1]
            for i, e in enumerate(cod):
                if i != 1+d:
                    p = p*computed_powers[e, i]
                elif e == 0:
                    p = p*0
                else:
                    p = p*computed_powers[e-1, i]
            if cod[0] != 0:
                p = p - functools.reduce(np.multiply, (computed_powers[c, i+1] for i, c in enumerate(cod[1:])),
                                         cod[0]*computed_powers[cod[0]-1, 0])
            # d1 = mn(e)* (e_1 x_0^{e_0}x_1^{e_1-1}x_2^{e_2}x_3^{e_3} - e_0x_0^{e_0-1}x_1^{e_1}x_2^{e_2}x_3^{e_3})
            dx_dy_dz[d, ci] = mn*p
    return dx_dy_dz


def basis_info(order, nsd, derivative=False, force_codec=None, printer=False):
    '''Bernstein Basis Utility'''
    import sympy
    pol, coeffs, basis = bernstein_space(order=order, nsd=nsd)
    codec = [list(map(int, str(c).split('_')[1:])) for c in coeffs]
    if force_codec is not None:
        if type(force_codec) is str:
            force_codec = codecs()[force_codec][2]
        reorder = np.lexsort(
            np.array(codec).T)[invert_permutation(np.lexsort(force_codec.T*order))]
        basis = [basis[b] for b in reorder]
        codec = [codec[c] for c in reorder]
        coeffs = [coeffs[s] for s in reorder]
    else:
        # this attempts to make corners in the front
        codec_argsort = sorted(
            enumerate(codec), key=lambda x: (-(np.array(x[1])**2).sum(), x[1][::-1]))
        codec = [codec[s] for s, _ in codec_argsort]
        coeffs = [coeffs[s] for s, _ in codec_argsort]
        basis = [basis[s] for s, _ in codec_argsort]
    points = create_point_set(codec)

    xyz =  [sympy.Symbol(f'x{i}') for i in range(nsd)]

    basis_wrapper = sympy.lambdify(xyz, basis)
    diff_wrapper = None
    if derivative:
        from sympy.utilities.autowrap import ufuncify
        diff_wrapper = [sympy.lambdify(
            xyz, [diff(l, i) for l in basis], 'numpy') for i in xyz]

    A = create_matrix(
        [pol.subs([(ix, p[i]) for i, ix in enumerate(xyz)]) for p in points], coeffs)
    b2l = np.asarray(A).astype(np.float64) #*order**(nsd+1)
    l2b = np.asarray(np.linalg.inv(b2l)).astype(np.float64) 
    result = dict(
        basis=basis_wrapper,
        basis_d=diff_wrapper,
        codec=codec,
        b2l=b2l, l2b=l2b
    )
    if printer:
        print(result)
        return None
    return result


def invert_permutation(p):
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def generate_jacobian(rule='wsj', fmt='.npz'):
    '''
    generate dxyz, for evaluation of Jacobians.
    '''
    import sympy
    import quadpy
    if rule == 'wsj':
        scheme = quadpy.tetrahedron.williams_shunn_jameson()
    if rule.startswith('uniform'):
        def scheme(_): return ''
        N = int(rule.split('uniform')[1])
        scheme.points = np.asarray(
            list(filter(lambda x: sum(x) == N, itertools.product(range(N+1), repeat=4))))/N
        scheme.weights = np.ones(len(scheme.points))/len(scheme.points)

    pol, coeffs, basis = bernstein_space(order=4, nsd=3)
    codec_op = np.array([tuple(map(int, str(c).split('_')[1:]))
                         for c in coeffs])

    tetra35_codec, tetra35_codec_n, tetra35_op = codecs()['tetra35']

    order_to_tetra = np.lexsort(
        codec_op.T)[invert_permutation(np.lexsort(tetra35_op.T))]
    basis = [basis[b] for b in order_to_tetra]
    coeffs = [coeffs[b] for b in order_to_tetra]

    var = [x, y, z]
    basis_diff = [numba.jit(sympy.lambdify(var,
                                           [diff(b, i) for b in basis]))
                  for i in var]

    dxyz = np.asarray([[basis_diff[dim](*pt)
                        for pt in tqdm.tqdm(scheme.points[:, 1:])] for dim in range(3)])

    points = create_point_set(tetra35_op)
    A = create_matrix([pol.subs([(x, p[0]), (y, p[1]), (z, p[2])])
                       for p in points], coeffs)

    lag_from_bern = np.asarray(A).astype(np.float64)
    bern_from_lag = np.linalg.inv(lag_from_bern)

    if fmt == 'npz':
        np.savez(f'curve/data/p4_q{rule}_dxyz.npz',
                 weights=scheme.weights,
                 points=scheme.points,
                 dxyz=dxyz@bern_from_lag,
                 notes='williams_shunn_jameson degree9, tetra35 order 4')
        print(
            f'saved to curve/data/p4_q{rule}_dxyz.npz with length {scheme.weights.shape}')
    elif fmt == 'h5':
        with h5py.File(f'curve/data/p4_q{rule}_dxyz.h5', 'w') as fp:
            fp['weights'] = scheme.weights
            fp['points'] = scheme.points
            fp['dxyz'] = dxyz@bern_from_lag
            fp['dxyz'].attrs['notes'] = ''
    else:
        print(f'{fmt} is not a valid format.')


def tensor_product_wedge18_gen(N=6):
    import sympy
    import itertools

    N1, N2 = N, N
    tri_samples = np.asarray(list(filter(lambda x: sum(
        x) == N1, itertools.product(range(N1+1), repeat=3))))/N1
    lin_samples = np.asarray(list(filter(lambda x: sum(
        x) == N2, itertools.product(range(N2+1), repeat=2))))/N2

    tri_data = basis_info(order=2, nsd=2, derivative=True)
    lin_data = basis_info(order=2, nsd=1, derivative=True)

    l_uv = np.asarray([tri_data['basis'](*p)
                       for p in tqdm.tqdm(tri_samples[:, 1:])])@tri_data['l2b']
    h_t = np.asarray([lin_data['basis'](*p)
                      for p in tqdm.tqdm(lin_samples[:, 1:])])@lin_data['l2b']
    l_duv = np.asarray([[tri_data['basis_d'][dim](*p)
                         for p in tqdm.tqdm(tri_samples[:, 1:])] for dim in range(2)])@tri_data['l2b']
    h_dt = np.asarray([lin_data['basis_d'][0](*p)
                       for p in tqdm.tqdm(lin_samples[:, 1:])])@lin_data['l2b']

    lh_duv = np.kron(l_duv, h_t)
    lh_dt = np.kron(l_uv, h_dt)[None, :]
    lh_diff = np.vstack([lh_duv, lh_dt])
    lh_uvt = np.kron(l_uv, h_t)

    tri_points = create_point_set(tri_data['codec'])
    lin_points = create_point_set(lin_data['codec'])
    from sympy.matrices import Matrix
    ts_points = np.asarray([Matrix.hstack(a, b)
                            for a, b in itertools.product(tri_points, lin_points)])
    wedge = codecs()['wedge18']

    reorder = np.lexsort(
        ts_points.T)[invert_permutation(np.lexsort(wedge[2].T))]

    samples = np.asarray([np.concatenate([a, b]) for a, b in itertools.product(
        tri_samples[:, 1:], lin_samples[:, 1:])])
    return lh_diff[:, :, reorder], ts_points.astype(np.float64)[reorder], samples


def test_tensor_product_gen():
    duvt, pts, samples = tensor_product_wedge18_gen(N=6)
    print(pts)
    def det(d): return (d[0]*np.cross(d[1], d[2])).sum(axis=1)

    # Constant
    assert np.linalg.norm(det(duvt@pts) - 1) < 1e-10
    # Scale
    assert np.linalg.norm(det(duvt@(2*pts + 3)) - 8) < 1e-10
    # Nonlinear
    assert np.linalg.norm(det(duvt@(pts**2)).min() - 0) < 1e-10
    # notice that max(u*v)=0.25, max(t)=1
    assert np.linalg.norm(det(duvt@(pts**2)).max() - 8*0.25*1) < 1e-10
    assert np.linalg.norm(
        (np.prod(samples, axis=1)*8 - det(duvt@(pts**2)))) < 1e-10

    assert np.abs(duvt.sum()) < 1e-10
    assert (duvt.max()-4) < 1e-10
    assert (duvt.min()+4) < 1e-10
    assert np.abs(duvt*27).sum() == 108116


def test_reorder():
    b = basis_info(2, 2, force_codec='tri6')
    print(b['codec'])


def test_derivative():
    b = basis_info(2, 2, force_codec='tri6', derivative=True)
    dx, dy = b['basis_d']
    data = np.ones((3, 2))
    print(np.array(dx(data[:, 0], data[:, 1])))


if __name__ == "__main__":
    import fire
    fire.Fire()
