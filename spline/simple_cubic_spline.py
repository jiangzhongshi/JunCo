#!/usr/bin/env python
"""Simple Means no Knot multiplicity etc. """
import scipy
import torch
import numpy as np
import numba
import sys
from sksparse.cholmod import cholesky_AAt

def cubic_bspline_polynomial_coeffs(derivative=0):
    # [x^0, x^1, x^2, x^3]
    # Mathematica `PiecewiseExpand[BSplineBasis[{3, {0, 1, 2, 3, 4}}, 0, x]]`
    # Spans 4 knot intervals.
    poly_coef = [[
        np.array([0, 0, 0, 1]) / 6,
        np.array([4, -12, 12, -3]) / 6,
        np.array([-44, 60, -24, 3]) / 6,
        np.array([64, -48, 12, -1]) / 6
    ]]

    def poly_coef_derivative(coef):
        """
        Take derivative of poly coefficients
        degree(coef) is len(j)-1
        """
        return [[np.arange(1, len(j)) * j[1:] for j in i] for i in coef]

    def flatten(l): return np.asarray([j for i in l for j in i])

    poly_list = [poly_coef]
    for _ in range(derivative):
        poly_list.append(poly_coef_derivative(poly_list[-1]))
    
    return (list(map(flatten, poly_list)))

poly_coefs  = cubic_bspline_polynomial_coeffs(2)

"""
This is a specific table construction. Each row stores the polynomial segments to use
# Left: 0 point for x, evaluated as f(x-Left). Always b_id -3
b_id: id of the basis, corresponding to control coefficent access.
Segment: from left to right, which of the segment is used here.
in the end, an additional last row is added for the purpose of evaluating on the last ending knot.
"""

def base_from_i(length, i):
    if i==length: # add last row to avoid numerical isssue
        i -= 1
    return [(i-j+3, j) for j in range(4)] # cp_id, segment

def table_1d(length):
    return [base_from_i(length, i) for i in range(length+1)]

def table_constructor(length, x):
    return [base_from_i(length, i) for i in x]


class BSplineSurface:
    def __init__(self, start, resolution, width, coef=None):
        self.start = np.asarray(start)
        self.scale = 1 / np.asarray(resolution)
        self.width = width # number of intervals
        self.coef = coef
        self.cache_factor = None
        self.TH, self.device = False, 'cpu'

    @staticmethod
    def _bspev_and_c(x, width, poly_coef):
        """BSpline evaluation, and coefficient ids.

        Args:
            x ([type]): [description]
            width ([type]): [description]
            poly_coef ([type]): [description]

        Returns:
            b[k]: the value of the basis at x[k].
            i: which coef id they correspond to
        """
        degree = poly_coef.shape[1]

        xi = np.floor(np.clip(x,0,width-0.1)).astype(np.int64)
        left = xi[:,None] - np.arange(4)[None,:]
        cid = np.tile(np.arange(4), (xi.shape[0],1))

        b = np.sum(poly_coef[cid] *
                  np.power((x[:, None] - left)[:, :, None],
                            np.arange(degree)),
                      axis=2)
        i = left + 3
        return b, i # there may be many zeros here to prune

    @staticmethod
    def _global_basis_row(x, width, scale, du=0, dv=0):
        bu, iu = BSplineSurface._bspev_and_c(x[:,0], width[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c(x[:,1], width[1], poly_coefs[dv])
        outer = np.einsum('bi,bo->bio', bu * (scale[0]**du), bv * (scale[1]**dv)).flatten()

        dim1 = (width[1]) + 3 # dim of controls, due to np rowmajor
        cols = (np.expand_dims(iu, 2) * dim1 + np.expand_dims(iv, 1)).flatten()
        rows = np.arange(iu.shape[0])[:, None].repeat(iu.shape[1]*iv.shape[1], axis=1).flatten()
        return rows, cols, outer


    @staticmethod
    def _global_basis_hessian_row(x, width, scale): # regularization in [Forsey and Wong 1998]
        row0, col0, data0 = BSplineSurface._global_basis_row(x, width, scale, 0, 2)
        row1, col1, data1 = BSplineSurface._global_basis_row(x, width, scale, 2, 0)
        row2, col2, data2 = BSplineSurface._global_basis_row(x, width, scale, 1, 1)
        row1 += row0.max()+1
        row2 += row1.max()+1
        return (np.concatenate([row0, row1, row2]),
                np.concatenate([col0, col1, col2]),
                np.concatenate([data0, data1, data2*np.sqrt(2)]))


    def ev(self, x, du=0, dv=0):
        x = self.transform(x)
        bu, iu = BSplineSurface._bspev_and_c(x[:,0], self.width[0], poly_coefs[du])
        bv, iv = BSplineSurface._bspev_and_c(x[:,1], self.width[1], poly_coefs[dv])

        coef_iuv = [c[(np.expand_dims(iu,2), np.expand_dims(iv,1))] for c in self.coef]
        bu *= self.scale[0]**(du)
        bv *= self.scale[1]**(dv)
        return np.hstack([np.einsum('bij,bjk,bk->bi', np.expand_dims(bu,1), c, bv) for c in coef_iuv])

    def get_global_basis_row_vec(self, X):
        return BSplineSurface._global_basis_row(self.transform(X), self.width, self.scale)

    def interpolate(self, X, f, regularize=True, cur_ev=None):
        X = self.transform(X)
        width = [w+1 for w in self.width]
        def add_half(l):
            return [0.5] + list(range(l)) #+ [l-0.5]
        num_reg = 4
        row0, col0, data0 = BSplineSurface._global_basis_row(X, self.width, self.scale)
        if regularize:
            regularizer = [[i,j] for i in add_half(num_reg*width[0]) for j in add_half(num_reg*width[1])]
            regularizer = np.array(regularizer)/num_reg

            row1, col1, data1 = BSplineSurface._global_basis_hessian_row(regularizer, self.width, self.scale)
            row1 += row0.max()+1
            reg_scale = 1e-5
            data1 *= reg_scale
            row0, col0, data0 = (np.concatenate([row0, row1]),
                                    np.concatenate([col0, col1]),
                                    np.concatenate([data0, data1]))
        A = scipy.sparse.csr_matrix((data0, (row0, col0)),
                                        shape=(row0.max()+1,
                                                (self.width[0]+3)*(self.width[1]+3)))

        print(A.shape)

        factor = cholesky_AAt(A.T)#, beta=1e-10)
        self_cache_factor = factor
        self_cache_At = A.T

        if regularize:
            reg = regularizer
            if cur_ev is None:
                reg_vec = np.zeros((reg.shape[0]*3,3))
            f2 = reg_scale * reg_vec
            f = np.vstack([f, f2])[:A.shape[0]]
        coef = self_cache_factor(A.T@f)
        res = (A @ coef - f)
        print('Residual', np.linalg.norm(res,axis=0))
        self.coef = [c.reshape(self.width[0]+3, self.width[1]+3) for c in coef.T]
        if self.TH:
            self.coef = [torch.from_numpy(c).to(self.device) for c in self.coef]
        return res[:X.shape[0]]

    def init_TH(self, cuda):
        """
        prepare for PyTorch
        """
        if self.TH:
            return
        self.TH = True
        self.coef = [torch.from_numpy(c) for c in self.coef]
        self.start, self.scale = map(torch.from_numpy, [self.start.astype(np.float64), self.scale])
        self.ev = self.ev_TH

        if cuda:
            self.start, self.scale = self.start.cuda(), self.scale.cuda()
            self.coef = [c.cuda() for c in self.coef]
            self.device = 'cuda'

    def untransform(self, y):
        return self.start + y / self.scale
    def transform(self, x):
        return (x - self.start) * self.scale


def mesh_coord(num):
    x = np.linspace(0, 1, num=num, endpoint=True)
    y = np.linspace(0, 1, num=num, endpoint=True)
    x, y = np.meshgrid(x, y)
    return np.vstack([x.ravel(), y.ravel()]).transpose()


# import quadpy
def fit(uv_fit, V, F, size, surf, filename=None):
    X = np.asarray([[i, j] for i in np.linspace(0, 1, size * 2) for j in np.linspace(0, 1, size * 2)])
    # fid, bc = utils.embree_project_face_bary(uv_fit, F,
                                            #  source=X, normals=None)
    invalid = np.where(fid == -1)[0]
    fid, bc, X = np.delete(fid, invalid), np.delete(bc, invalid, axis=0), np.delete(X, invalid, axis=0)
    Z = np.einsum('bi, bij->bj', bc, V[F[fid]])

    for _ in range(1):
        uv_fit, _ = igl.upsample(uv_fit, F)
        V, _ = igl.upsample(V,F)
    print('Upsampled to', V.shape)
    X = np.vstack([X, uv_fit, uv_fit[F].mean(axis=1)])
    Z = np.vstack([Z, V, V[F].mean(axis=1)])

    surf.interpolate(X, Z)
    return surf

if __name__ == '__main__':
    size = 4
    cbs = BSplineSurface(start=[0, 0],
                        resolution=[1 / size, 2 / size],
                        width=[size, size], coef=None)
    cbs.interpolate(uv_fit,V)
