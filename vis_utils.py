import numpy as np
import igl
import meshplot as mp
import h5py
import plotly.graph_objects as go
import pythreejs as p3s
import scipy
from sidecar import Sidecar


from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

def h5deser(filename):
    with h5py.File(filename,'r') as f:
        F, V, base, top = f['mF'][()], f['mV'][()], f['mbase'][()], f['mtop'][()]
        refV, refF = f['ref.V'][()],f['ref.F'][()]
        grouped_tracker = np.split(f['track_flat'][()].astype(np.int32).flatten(), np.cumsum(f['track_size'][()]).astype(np.int))
    return refV, refF, base, top, V,F, grouped_tracker[:-1]

def mips2d(V,f):
    e1 = V[f[1]] - V[f[0]]
    e2 = V[f[2]] - V[f[0]]
    e1_len = np.linalg.norm(e1)
    e2_x = np.dot(e1,e2) / e1_len
    e2_y = np.linalg.norm(e2 - e2_x * e1 / e1_len)
    tri = np.array([[e1_len, e2_x], [0, e2_y]])
    ref = np.array([[1,0.5],[0, np.sqrt(3)/2]])
    jac = tri@np.linalg.inv(ref)
    invf2 = np.sum(np.linalg.inv(jac)**2)
    frob2 = np.sum(jac**2)
    return frob2*invf2

def centered(V):
    V -= V.min(axis=0)
    V /= V.max()
    return V

def stack_V_F(Vl,Fl):
    v_cnt = 0
    for v,f in zip(Vl,Fl):
        f[:] += v_cnt
        v_cnt += v.shape[0]
    return np.vstack(Vl), np.vstack(Fl)

def smoother(L, b, V):
    mask = np.zeros(L.shape[0])
    mask[b] = 1
    Lii = L[mask==0][:,mask==0]
    Lib = L[mask==0][:,mask==1]
    V = V.copy()
    print(np.linalg.norm(L@V))
    
    V[mask==0] = scipy.sparse.linalg.spsolve(-Lii, Lib@V[mask==1])
    print(np.linalg.norm(L@V))
    return V


def doo_sabin(V,F, eps=0.05):
    dsF = []
    v_num = V.shape[0]
    f_num = F.shape[0]
    he_num = F.shape[0]*3
    # total v_num + he_num (==3*f_num) + 3*f_num
    dsV = np.zeros((v_num + he_num + 3*f_num,3))
    dsV[:v_num] = V
    FF, FFi = igl.triangle_triangle_adjacency(F)
    for i in range(F.shape[0]):
        v0,v1,v2 = F[i]
        f0, f1, f2 = v_num + he_num + 3*i + np.arange(3)
        e0 = v_num + 3*i+np.arange(3)
        e1 = np.zeros(3)
        for e in range(3):
            f_oppo, e_oppo = FF[i,e], FFi[i,e]
            e1[e] = v_num + 3*f_oppo + e_oppo
        newF = np.array([[v0, e0[0], f0],
                        [v0, f0, e1[2]],
                        [e0[0],e1[0],f1],
                        [e0[0],f1,f0],
                        [v1,f1,e1[0]],
                        [v1, e0[1],f1],
                        [e0[1],f2,f1],
                        [e0[1],e1[1],f2],
                        [e1[1],v2,f2],
                         [f2,v2,e0[2]],
                         [f2,e0[2],f0],
                         [f0,e0[2],e1[2]],
                        [f0,f1,f2]])
        dsF.append(newF)
        fx = [f0,f1,f2]
        for e in range(3):
            dsV[e0[e]] = (1-eps)*V[F[i,e]] + eps * V[F[i,(e+1)%3]]
            dsV[fx[e]] = (1-eps)*V[F[i,e]] + eps/2 * V[F[i,(e+1)%3]] + eps/2*V[F[i,(e+2)%3]]
    return dsV, np.array(dsF)

from matplotlib import cm

def triangle_quality(p0,p1,p2):
    e1, e2 = p1 - p0, p2 - p0
    e1_len = np.linalg.norm(e1)
    e2_x = e1.dot(e2) / e1_len
    e2_y = np.linalg.norm(e2 - e2_x * e1 / e1_len)
    tri = np.array([[e1_len, e2_x],[0, e2_y]])
    ref = np.array([[1, 0.5],[0,  np.sqrt(3) / 2]])
    jac = tri@np.linalg.inv(ref)
    frob2 = (jac**2).sum()
    det = np.linalg.det(jac)
    return frob2/det

def msubplot(v_list,f_list,shape ,**sh):
    plt = None
    for i,(v,f) in enumerate(zip(v_list,f_list)):
        vw = mp.Viewer(sh)
        vw.add_mesh(v,f,shading=sh)
        plt = mp.Subplot(plt, vw, s=[shape[0],shape[1],i])
    return plt


def str_to_array(s):
    return np.array(list(map(float,s.split())))


def f2e(F):
    return np.vstack([F[:,[0,2]],F[:,[0,1]],F[:,[1,2]]])

def t2e(T):
    return np.vstack([T[:,e] for e in (itertools.combinations(range(T.shape[1]),2))])

def h5reader(file, *names):
    with h5py.File(file,'r') as fp:
        if len(names) == 0:
            return str(fp.keys())
        a = list(map(lambda x:fp[x][()], names))
    return a


def add_transparent_mesh(self, v, f, c=None, uv=None, n=None, shading={}, opacity=0.6):
    import pythreejs as p3s
    sh = self._Viewer__get_shading(shading)
    mesh_obj = {}

    #it is a tet
    if v.shape[1] == 3 and f.shape[1] == 4:
        f_tmp = np.ndarray([f.shape[0]*4, 3], dtype=f.dtype)
        for i in range(f.shape[0]):
            f_tmp[i*4+0] = np.array([f[i][1], f[i][0], f[i][2]])
            f_tmp[i*4+1] = np.array([f[i][0], f[i][1], f[i][3]])
            f_tmp[i*4+2] = np.array([f[i][1], f[i][2], f[i][3]])
            f_tmp[i*4+3] = np.array([f[i][2], f[i][0], f[i][3]])
        f = f_tmp

    if v.shape[1] == 2:
        v = np.append(v, np.zeros([v.shape[0], 1]), 1)


    # Type adjustment vertices
    v = v.astype("float32", copy=False)

    # Color setup
    colors, coloring = self._Viewer__get_colors(v, f, c, sh)

    # Type adjustment faces and colors
    c = colors.astype("float32", copy=False)

    # Material and geometry setup
    ba_dict = {"color": p3s.BufferAttribute(c)}
    if coloring == "FaceColors":
        verts = np.zeros((f.shape[0]*3, 3), dtype="float32")
        for ii in range(f.shape[0]):
            #print(ii*3, f[ii])
            verts[ii*3] = v[f[ii,0]]
            verts[ii*3+1] = v[f[ii,1]]
            verts[ii*3+2] = v[f[ii,2]]
        v = verts
    else:
        f = f.astype("uint32", copy=False).ravel()
        ba_dict["index"] = p3s.BufferAttribute(f, normalized=False)

    ba_dict["position"] = p3s.BufferAttribute(v, normalized=False)

    if uv is not None:
        uv = (uv - np.min(uv)) / (np.max(uv) - np.min(uv))
        # tex = p3s.DataTexture(data=texture_data, format="RGBFormat", type="FloatType")
        material = p3s.MeshStandardMaterial(map=texture_data, reflectivity=sh["reflectivity"], side=sh["side"],
                roughness=sh["roughness"], metalness=sh["metalness"], flatShading=sh["flat"],
                polygonOffset=True, polygonOffsetFactor= 1, polygonOffsetUnits=5)
        ba_dict["uv"] = p3s.BufferAttribute(uv.astype("float32", copy=False))
    else:
        material = p3s.MeshStandardMaterial(vertexColors=coloring, reflectivity=sh["reflectivity"],
                    side=sh["side"], roughness=sh["roughness"], metalness=sh["metalness"], 
                                            opacity=opacity, transparent=True,alphaTest=opacity*0.99,
                                            blending='CustomBlending',depthWrite=False,
                    flatShading=True)

    if type(n) != type(None) and coloring == "VertexColors":
        ba_dict["normal"] = p3s.BufferAttribute(n.astype("float32", copy=False), normalized=True)

    geometry = p3s.BufferGeometry(attributes=ba_dict)

    if coloring == "VertexColors" and type(n) == type(None):
        geometry.exec_three_obj_method('computeVertexNormals')
    elif coloring == "FaceColors" and type(n) == type(None):
        geometry.exec_three_obj_method('computeFaceNormals')

    # Mesh setup
    mesh = p3s.Mesh(geometry=geometry, material=material)

    # Wireframe setup
    mesh_obj["wireframe"] = None

    # Object setup
    mesh_obj["max"] = np.max(v, axis=0)
    mesh_obj["min"] = np.min(v, axis=0)
    mesh_obj["geometry"] = geometry
    mesh_obj["mesh"] = mesh
    mesh_obj["material"] = material
    mesh_obj["type"] = "Mesh"
    mesh_obj["shading"] = sh
    mesh_obj["coloring"] = coloring

    return self._Viewer__add_object(mesh_obj)

def add_matcap_mesh(self, v, f, c=None, uv=None, n=None, shading={}, texture_data=None):
    sh = self._Viewer__get_shading(shading)
    mesh_obj = {}

    #it is a tet
    if v.shape[1] == 3 and f.shape[1] == 4:
        f_tmp = np.ndarray([f.shape[0]*4, 3], dtype=f.dtype)
        for i in range(f.shape[0]):
            f_tmp[i*4+0] = np.array([f[i][1], f[i][0], f[i][2]])
            f_tmp[i*4+1] = np.array([f[i][0], f[i][1], f[i][3]])
            f_tmp[i*4+2] = np.array([f[i][1], f[i][2], f[i][3]])
            f_tmp[i*4+3] = np.array([f[i][2], f[i][0], f[i][3]])
        f = f_tmp

    if v.shape[1] == 2:
        v = np.append(v, np.zeros([v.shape[0], 1]), 1)


    # Type adjustment vertices
    v = v.astype("float32", copy=False)

    # Color setup
    colors, coloring = self._Viewer__get_colors(v, f, c, sh)

    # Type adjustment faces and colors
    c = colors.astype("float32", copy=False)

    # Material and geometry setup
    ba_dict = {"color": p3s.BufferAttribute(c)}
    if coloring == "FaceColors":
        verts = np.zeros((f.shape[0]*3, 3), dtype="float32")
        for ii in range(f.shape[0]):
            #print(ii*3, f[ii])
            verts[ii*3] = v[f[ii,0]]
            verts[ii*3+1] = v[f[ii,1]]
            verts[ii*3+2] = v[f[ii,2]]
        v = verts
    else:
        f = f.astype("uint32", copy=False).ravel()
        ba_dict["index"] = p3s.BufferAttribute(f, normalized=False)

    ba_dict["position"] = p3s.BufferAttribute(v, normalized=False)

    if type(uv) != type(None):
        uv = (uv - np.min(uv)) / (np.max(uv) - np.min(uv))
        # tex = p3s.DataTexture(data=texture_data, format="RGBFormat", type="FloatType")
        material = p3s.MeshStandardMaterial(map=texture_data, reflectivity=sh["reflectivity"], side=sh["side"],
                roughness=sh["roughness"], metalness=sh["metalness"], flatShading=sh["flat"],
                polygonOffset=True, polygonOffsetFactor= 1, polygonOffsetUnits=5)
        ba_dict["uv"] = p3s.BufferAttribute(uv.astype("float32", copy=False))
    else:
        material = p3s.MeshMatcapMaterial(vertexColors=coloring, 
                                            reflectivity=sh["reflectivity"],
                                            side=sh["side"],
                                            roughness=sh["roughness"], 
                                            metalness=sh["metalness"],
                                            flatShading=sh["flat"],
                                            map = texture_data,
                                            matcap = texture_data,
                polygonOffset=True, polygonOffsetFactor= 1, polygonOffsetUnits=5)

    if type(n) != type(None) and coloring == "VertexColors":
        ba_dict["normal"] = p3s.BufferAttribute(n.astype("float32", copy=False), normalized=True)

    geometry = p3s.BufferGeometry(attributes=ba_dict)

    if coloring == "VertexColors" and type(n) == type(None):
        geometry.exec_three_obj_method('computeVertexNormals')
    elif coloring == "FaceColors" and type(n) == type(None):
        geometry.exec_three_obj_method('computeFaceNormals')

    # Mesh setup
    mesh = p3s.Mesh(geometry=geometry, material=material)

    # Wireframe setup
    mesh_obj["wireframe"] = None

    # Object setup
    mesh_obj["max"] = np.max(v, axis=0)
    mesh_obj["min"] = np.min(v, axis=0)
    mesh_obj["geometry"] = geometry
    mesh_obj["mesh"] = mesh
    mesh_obj["material"] = material
    mesh_obj["type"] = "Mesh"
    mesh_obj["shading"] = sh
    mesh_obj["coloring"] = coloring

    return self._Viewer__add_object(mesh_obj)


def obj_writer():
    with open('temp.obj','w+') as fp:
        fp.writelines(['v ' + ' '.join(map(str,list(v)))+'\n' for v in SV])
        fp.writelines(['vt ' + ' '.join(map(str,list(v)))+'\n' for v in UV])
        fp.writelines([f'f {f[0]}/{tf[0]} {f[1]}/{tf[1]}  {f[2]}/{tf[2]}\n' for f,tf in zip(SF+1,TF+1)])


def ply_writer(filename, uV,uF,uC):
    vertex=np.array([tuple(list(v)+ list(c)) for v,c in zip(uV,uC)],
                dtype=[('x', 'f4'), ('y', 'f4'),('z','f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    face = np.array([tuple([f]) for f in uF],dtype=[('vertex_indices', 'i4', (3,))])
    from plyfile import PlyData, PlyElement
    vel = PlyElement.describe(vertex, 'vertex')
    fel = PlyElement.describe(face, 'face')

    PlyData([vel,fel]).write(filename)

def scale(x):
    scale.a = x.min(axis=0)
    y = x-scale.a
    scale.b = y.max()
    y = y/scale.b
    return y
use_scale=lambda x:(x-scale.a)/scale.b


def matcap_checkers(n_checkers_x, n_checkers_y, width=256, height=256):
    import pythreejs as p3s
    # tex dims need to be power of two.
    array = np.ones((width, height, 3), dtype='float32')

    # width in texels of each checker
    checker_w = width / n_checkers_x
    checker_h = height / n_checkers_y

    for y in range(height):
        for x in range(width):
            color_key = int(x / checker_w) + int(y / checker_h)
            if color_key % 2 == 0:
                array[x, y, :] = [0.9, 0.9, 0.2]
            else:
                array[x, y, :] = [0.2, 0.2, 0.2]
    return p3s.DataTexture(array, format="RGBFormat", type="FloatType")


def plot_mesh_and_checker(V,F,E, g_tex=None):
    '''two windows, left reflection lines, right pure shading. gtex is default to (60,1)
    '''
    if g_tex is None:
        g_tex = matcap_checkers(60,1)
    vw0=mp.Viewer(dict(height=1000,width=1000))
    add_matcap_mesh(vw0,V,F,texture_data=g_tex,shading=dict(flat=False))
    vw1=mp.Viewer(dict(height=1000,width=1000))
    vw1.add_mesh(V,F,shading=dict(flat=False))
    if E is not None: vw1.add_edges(V,E)
    plt = mp.Subplot(None, vw0, s=[1,2,0])
    plt = mp.Subplot(plt, vw1, s=[1,2,1])
    return plt

import itertools
import numpy as np
def twelve_tetra_table():
    V = np.array([[0,0,0],
                  [1,0,0],
                  [0,1,0],
                  [0,0,1],
                  [1,0,1],
                  [0,1,1]
                  ])
    twelve = []
    for tet in set(itertools.combinations(range(6),4)):
        tet = np.array(tet)
        vol = np.linalg.det(V[tet[1:]]-V[tet[0]])
        if vol == 0.0:
            continue
        if vol < 0:
            tet[2:] = tet[-1], tet[-2]
        twelve.append(tet)
    return np.array(twelve)


def sync_camera(plt, plt0):
    '''empirical snippet to assign camera of plt0 to plt'''
    plt._cam.position = plt0._cam.position
    plt._orbit.exec_three_obj_method('update')
    plt._cam.exec_three_obj_method('updateProjectionMatrix')
    
    
def corner_to_cube(c0, c1):
    mi, ma = np.minimum(c0,c1), np.maximum(c0,c1)
    pts = np.array([[mi[i] if tup[i] else ma[i] for i in range(3)]
                    for tup in itertools.product([False,True],repeat=3)])
    tris = np.array([[0,1,2],[1,2,3],[4,5,6],[6,7,5],
                    [0,2,4],[2,6,4],[3,7,5],[3,1,5]])
    return pts, tris

def trackee_debug_snippet():
    '''just save some snippet for debugging trackee with degree-1 terminal.'''
    mF,mV, mB, mT, refF, refV, meta_f, meta_i,track_f,track_s = h5reader('../build_clang/block_curve.h5', 
                                                                            'mF', 'mV', 'mbase', 'mtop', 
                                      'ref.F','ref.V','meta_edges_flat','meta_edges_ind','track_flat','track_size')
    tracks = np.split(track_f.astype(np.int32).flatten(), np.cumsum(track_s.astype(np.int)))
    metas = np.split(meta_f, meta_i[1:-1])

    chain_zero = np.array([i[:2] for i in metas if i[2] ==0 ])
    dict(zip(*np.unique(chain_zero, return_counts=True)))
    
    for fid,_ in [(i,f) for i,f in enumerate(mF) if 81 in f]:
        plt = mplot(mV,chain_zero,width=800,height=800)
        plt.add_points(mV[chain_zero.flatten()],shading=dict(point_size=0.1),c=np.ones(len(chain_zero.flatten())))
        plt.add_points(mV[mF[fid]],shading=dict(point_size=0.06))
        plt.add_mesh(refV,refF[tracks[fid]],c=np.arange(len(tracks[fid])),shading=dict(wireframe=True))
        
        
def shrink(tetV,tetT, alpha):
    VT = tetV[tetT]
    mean = VT.mean(axis=1,keepdims=True)
    return (VT - mean)*alpha + mean
    # mp.plot(shrink(V,T,0.7).reshape(-1,3),np.arange(4*len(T)).reshape(-1,4))
    
    
def find_correspondence():
    '''save some scripts for reuse'''
    import seism

    pc = seism.PrismCage('../build_clang/sculpt_curve.h5')
    inpV,_ = h5reader('../build_clang/sculpt.h5','ref.V','ref.F')
    flat_sv = np.einsum('fed,es->fsd',
                        np.einsum('fDd,eD->fed',mV[mF], np.array(tri10['codec']))/3,
                        bas_val)
    pc_fid, pc_uv = pc.transfer(refV, refF,flat_sv.reshape(-1,3))

    sampled_pos = np.einsum('fed,fe->fd',inpV[refF[pc_fid.astype(np.int)]], np.hstack([1-pc_uv.sum(axis=1, keepdims=True), pc_uv]))

    trans_uv = np.einsum('fed,fe->fd',inpV[refF[pc_fid.astype(np.int)], :2], np.hstack([1-pc_uv.sum(axis=1, keepdims=True), pc_uv]))
    
def find_phong_projections():
    import seism
    tree = seism.AABB(refV,refF)
    flat_bot = np.einsum('fDd,eD->fed', 
                                  mB[mF], 
                                  np.hstack([1-usV.sum(axis=1,keepdims=True),
              usV]))
    flat_top = np.einsum('fDd,eD->fed', 
                                  mT[mF], 
                                  np.hstack([1-usV.sum(axis=1,keepdims=True),
              usV]))

    queries = np.array([tree.segment_hit(b, t,
                                False) for b,t in zip(tqdm.tqdm(flat_bot.reshape(-1,3)), flat_top.reshape(-1,3))])


    pc_fid, pc_uv = queries[:,0], queries[:,1:-1]

    #pc_fid, pc_uv = pc.transfer(refV, refF, flat_sv.reshape(-1,3))

    sampled_pos = np.einsum('fed,fe->fd',
                            inpV[refF[pc_fid.astype(np.int)]], 
                            np.hstack([1-pc_uv.sum(axis=1, keepdims=True), pc_uv]))
    
def tetmesh_from_shell(base, top, F):
    tetra_splits = (np.array([0, 3, 4, 5, 1, 4, 2, 0, 2, 5, 0, 4]).reshape(-1, 4),
                    np.array([0, 3, 4, 5, 1, 4, 5, 0, 2, 5, 0, 1]).reshape(-1, 4))
    vnum = len(base)
    T = []
    for f in F:
        tet_c = tetra_splits[0] if f[1] > f[2] else tetra_splits[1]
        T.append((tet_c // 3)*vnum + f[tet_c % 3])
    return np.vstack([base, top]), np.vstack(T)

def control_points_duplicate_consistent():
    VF,NI = igl.vertex_triangle_adjacency(mF, len(mV))
    VFs = np.split(VF,NI[1:-1])
    for v, nbF in enumerate(VFs):
        vals = [cp[f, np.argmax(mF[f] == v)] for f in nbF]
        if sum((v-vals[0]).sum()>0 for v in vals) > 0:
            print(v)

def highorder_sv(cp,level=3, order=3):
    import sys
    sys.path.append('../python')
    from curve import fem_generator
    def local_upsample(level:int):
        usV, usF = igl.upsample(np.eye(3)[:,1:], np.arange(3)[None,:], level)
        bnd0 = igl.boundary_loop(usF)
        usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE
    usV,usF,usE = local_upsample(level=level)
    if order is None:
        size_to_order = lambda x: int((np.sqrt(x*8+1) - 1)/2) - 1
        order = size_to_order(cp.shape[1])
    bas_val = fem_generator.bernstein_evaluator(usV[:,0],usV[:,1],usV[:,0]*0,
                                                fem_generator.tuple_gen(order=order,var_n=2)).T
    sv = (bas_val@cp)
    return sv, np.vstack([usF+i*len(usV) for i in range(len(sv))]), np.vstack([usE+i*len(usV) for i in range(len(sv))])                  
    
def reorder_tetra():
    import sys
    sys.path.append('../python')

    from curve import fem_generator

    gmsh_cod = (fem_generator.codecs()['tetra35'][-1]*4).astype(np.int)

    auto_cod = fem_generator.tuple_gen(order=4, var_n=3)

    reorder = np.lexsort(
            np.array(auto_cod).T)[fem_generator.invert_permutation(np.lexsort(gmsh_cod.T))]

    assert np.all(np.array(auto_cod)== gmsh_cod[fem_generator.invert_permutation(reorder)])

    import h5py

    m = meshio.read('/home/zhongshi/public//curved/block.msh')

    with h5py.File('../build_clang/block_msh_autocod.h5', 'w') as fp:
        fp['lagr'] = m.points
        fp['cells'] = m.cells[0].data[:,fem_generator.invert_permutation(reorder)]

        
def tw_mips3d(V,T):
    from numpy import sqrt
    Jacs = V[T[:,1:]]-V[T[:,:1]]
    Jacs = np.linalg.inv(np.array([[1,0,0],[0.5,sqrt(3)/2.,0],[0.5,sqrt(3)/6.,sqrt(2)/sqrt(3)]]))@Jacs
    frob = np.sum(Jacs**2,axis=1).sum(axis=1)
    dets = np.linalg.det(Jacs)
    return frob/dets**(2/3)


def experiment_with_index_setting_on_split_triple_tetra():
    """Can we split triple tetra prisms and hope the subprisms to be still valid?"""
    V = np.array([[0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,0,1],
                      [1,0,1],
                      [0,1,1]
                      ]).astype(np.float)

    tetra_splits = (np.array([0, 3, 4, 5, 1, 4, 2, 0, 2, 5, 0, 4]).reshape(-1, 4),
                        np.array([0, 3, 4, 5, 1, 4, 5, 0, 2, 5, 0, 1]).reshape(-1, 4))

    T= twelve_tetra_table()

    '''Run this to get a valid prism to try'''
    for _ in range(1000):
        V1 = np.random.rand(6,3)
        base = np.array([[0,1,2,3],[0,1,2,4],[0,1,2,5]])
        top = np.array([[0,3,4,5],[1,3,4,5],[2,3,4,5]])
        if (igl.volume(V1,np.vstack([tetra_splits[1],
                                     base,top
                                    ])).min() > 0):
            print(V1)
            break

    '''Index is the edge to split, and g is where this new vertex is placed. It seems that very easily we get to a all negative situation'''
    for index in range(3):
        print(index)
        Vz,Tz = np.zeros((7,3)), np.zeros((7,3))
        Vz[1::2], Tz[1::2] = V1[:3], V1[3:]
        Vz[::2] = (V1[(index+1)%3] + V1[index])/2
        Tz[::2] = (V1[(index+1)%3+3] + V1[index+3])/2
        vid = np.roll([1,3,5],-index)
        for g in range(0,7,2):
            f = np.array([[vid[2],vid[0],g], [vid[2],g,vid[1]]])
            f = [np.roll(x,-x.argmin()) for x in f]
            print(igl.volume(*tetmesh_from_shell(Vz,Tz,f)).min())
            

def vtk_hex_read():
    m = meshio.read('3648.hh.sat.vtk')

    hexface =  np.array([[0,1,2],[0,2,3],
                       [0,3,7],[0,7,4],
                       [0,1,4],[1,4,5],
                       [1,2,5],[2,5,6],
                       [2,3,6],[3,6,7],
                       [7,6,5],[5,4,7]])
    hexedge = np.array([[0,4],[1,5],
                          [2,6],[3,7],
                         [0,1],[1,2],[2,3],[0,3],
                         [4,5],[5,6],[6,7],[4,7]])

    Verts = m.points[m.cells[0].data].reshape(-1,3)
    p = mp.plot(Verts,
             np.vstack([hexface+i*8 for i in range(len(m.cells[0].data))])
              ,wireframe=False,width=1600,height=1600)
    p.add_edges(Verts,
                np.vstack([hexedge+i*8 for i in range(len(m.cells[0].data))]))
    

def check_feature_ear(F,E):
    Eset = set(tuple(sorted(t)) for t in E)

    for f in F:
        s = sum([tuple(sorted([f[j], f[j-2]])) in Eset for j in range(3)])
        if s >= 2:
            print(f)
        if s == 3:
            print('33', f)
            
            
def glue_union_chain(segs):
    search = {t[0]:list(t) for t in segs}
    x = next(iter(search))
    while len(search) >= 2:
        l = search[x]
        nl = search.pop(l[-1], None)
        if nl is None:# reaching the end
            it = iter(search)
            x1 = next(it)
            while x1 == x:
                x1 = next(it)
            x = x1
        else:
            l+=(nl[1:])
    return search[x]


def mark_feature(V,F,deg):
    FN = igl.per_face_normals(V,F,np.ones(3))
    TT,TTi = igl.triangle_triangle_adjacency(F)
    di_angles = (FN[TT]*FN[:,None,:]).sum(axis=2)
    E =np.array([(F[f,e], F[f,e-2]) for f,e in zip(*np.where(di_angles <np.cos(np.deg2rad(180-deg))))])
    return E

def poisson_example():
    import point_cloud_utils as pcu
    sample = 2000
    fid, uv = pcu.sample_mesh_poisson_disk(refV,refF,sample)
    with h5py.File(f'../buildr/bunny.pp{sample}.init','a') as fp:
        fp['Poisson_f'] = fid
        fp['Poisson_uv'] = uv
        fp['Poisson_pts'] = np.sum(refV[refF[fid]]*uv[:,:,None],axis=1)

    Pp = h5reader('../tests/data/bunny.pp200.init','Poisson_pts')[0]
    cp, mF = h5reader('../tests/data/bunny.pp200.h5', 'complete_cp','mF')
    import meshzoo
    osV, osF = meshzoo.octa_sphere(10)
    igl.write_triangle_mesh(f'../tests/data/bunny/pp{iden}.obj', np.vstack([osV*5e-3+p for p in Pp]),np.vstack([osF+len(osV)*i for i in range(len(Pp))]))
    return 


def write_obj_lines(filename, V,E):
    with open(filename,'w') as fp:
        fp.write('\n'.join([f'v {v[0]} {v[1]} {v[2]}' for v in V] + 
                           [f'l {e[0]} {e[1]}' for e in E+1]))
        
def convert_tri10():
    import sys
    sys.path.append('../python/')

    from curve import fem_generator
    gmshcod = (fem_generator.codecs()['tri10'][2]*3).astype(np.int)
    autocod = fem_generator.tuple_gen(order=3,var_n=2)
    reorder = np.lexsort(
                np.array(autocod).T)[fem_generator.invert_permutation(np.lexsort(gmshcod.T))]
    np.all(np.array(autocod)[reorder] == gmshcod)

    tri10info = fem_generator.basis_info(order=3,nsd=2)

    def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i]*j]

    auto_cod_n = np.array([codec_to_n(c) for c in autocod])

    uniq_tup, tup_ind, tup_inv = np.unique(np.sort(mF[:,auto_cod_n].reshape(-1,3)), axis=0,return_index=True, return_inverse=True)

    m = meshio.Mesh(points=(tri10info['b2l']@scale_cp).reshape(-1,3)[tup_ind], cells = [('triangle10', 
                                                  tup_inv[np.arange(len(cp)*10).reshape(-1,10)][:,reorder]
                                                                      )])
    

def tet_highorder_sv(cp,level=3, order=3):
    dim= 3
    def local_upsample(level:int):
        tet = igl.boundary_facets(np.array([0, 1, 2, 3]).reshape(1,4))
        vv = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        usV, usF = igl.upsample(vv, tet, level)
        FN = igl.per_face_normals(usV,usF,np.ones(3))
        TT,TTi = igl.triangle_triangle_adjacency(usF)
        di_angles = (FN[TT]*FN[:,None,:]).sum(axis=2)
        usE =np.array([(usF[f,e], usF[f,e-2]) for f,e in zip(*np.where(di_angles < 0.5))])
        #bnd0 = igl.boundary_loop(usF)
        #usE = np.vstack([bnd0, np.roll(bnd0, -1)]).T
        return usV, usF, usE
    usV,usF,usE = local_upsample(level=level)
    #tuples = fem_generator.tuple_gen(order=order, var_n=dim)
    info = fem_generator.basis_info(order, dim, force_codec='tetra20')
    l2b = info['l2b']
    bas_val = fem_generator.bernstein_evaluator(usV[:,0],usV[:,1],usV[:,2], info['codec']).T
    sv = (bas_val@(l2b@cp))
    return sv, np.vstack([usF+i*len(usV) for i in range(len(sv))]), np.vstack([usE+i*len(usV) for i in range(len(sv))])