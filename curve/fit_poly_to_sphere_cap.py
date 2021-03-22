import numpy as np
import scipy
import scipy.interpolate
from scipy.interpolate import lagrange
from plotly import graph_objects as go
from scipy.special import erf
def test_2d(r, d=3):
    '''Check the interpolation error of a circular arc with different degrees of polynomials'''
    center_y = -np.sqrt(1-r**2,dtype=np.float128)
    h = 1 + center_y
    xi = np.linspace(-r,r,d+1,dtype=np.float128)
   # yi = np.cos(xi)
    yi = np.sqrt(1-xi**2) + center_y
    f = lagrange(xi, yi)
    x0 = np.linspace(-r,r,100)
    pts = np.vstack([x0,f(x0)]).T
    center = np.array([0,center_y])
    return np.abs(np.linalg.norm(pts - center, axis=1)
                  -1).max()

if __name__ == '__main__':
    y = [[test_2d(x/np.sqrt(3,dtype=np.float128),d=d) for x in np.linspace(1e-3,1,1000)] for d in range(1,10)]

    go.Figure([go.Scatter(x=np.linspace(1e-3,1,1000).astype(float),
                        y=np.log10(y[i]).astype(np.float),name=d) for i,d in enumerate(range(1,10))]
            ).update_xaxes(type='log')#.update_yaxes(type='log',tickformat='.0e')