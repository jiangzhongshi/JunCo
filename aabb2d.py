# +
import igl
import meshplot as mp
import plotly
import plotly.graph_objects as go
import numpy as np
import scipy

P = np.load('camelcurve.npy')

numP = P.shape[0]

binary_tree = [None]*(numP*2-1)
def buildBox(Points):
    return (Points.min(axis=0), Points.max(axis=0))
def build_AABB(Points, node):
    global binary_tree
    print(Points.shape, node)
    num_seg = len(Points)-1
    binary_tree[node] = buildBox(Points)
    if num_seg == 1:
        return
    assert num_seg > 1
    
    build_AABB(Points[:num_seg//2+1], node*2+1)
    build_AABB(Points[num_seg//2:], node*2+2)
build_AABB(np.vstack([P, P[0]]),0)

import igl
import meshplot as mp
import plotly
import plotly.graph_objects as go
import numpy as np
import scipy

P = np.load('camelcurve.npy')

numP = P.shape[0]
binary_tree = [None]*(2**(np.ceil(np.log(numP)/np.log(2)).astype(int)+1))
def buildBox(Points):
    return (Points.min(axis=0), Points.max(axis=0))
def build_AABB(Points, node):
    global binary_tree
    num_seg = len(Points)-1
    binary_tree[node] = buildBox(Points)
    if num_seg == 1:
        return
    assert num_seg > 1
    
    build_AABB(Points[:num_seg//2+1], node*2+1)
    build_AABB(Points[num_seg//2:], node*2+2)
build_AABB(np.vstack([P, P[0]]),0)

def box_corners(bb):
    x0, y0 = bb[0]
    x1, y1 = bb[1]
    return np.asarray([[x0,y0], [x0,y1],[x1,y0],[x1,y1]])

def per_point_normals(P):
    seg_n = np.zeros((len(P),2))
    for i, (v0,v1) in enumerate(zip(P, np.vstack([P[1:],P[0]]))):
        d = (v1-v0)
        seg_n[i] = (-d[1], d[0])
    normal = seg_n + np.roll(seg_n,axis=0,shift=1)
    return normal/np.linalg.norm(normal,axis=1)[:,None]
N = per_point_normals(P)

plt = mp.plot(np.vstack([P, P[0]]), np.vstack([np.arange(1,numP+1),np.arange(numP)]).T,shading=dict(point_size=0.1, line_color='red'), return_plot=True)
l=7
for i in range(2**l-1, 2**(l+1)-1):
    plt.add_edges(box_corners(binary_tree[i]), np.array([[0,1],[1,3],[2,3],[2,0]]))
