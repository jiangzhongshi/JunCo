### Misc utilities to improve meshplot

import numpy as np
import igl
import meshplot as mp
import h5py
import plotly.graph_objects as go
import pythreejs as p3s
import scipy


def f2e(F):
    return np.vstack([F[:,[0,2]],F[:,[0,1]],F[:,[1,2]]])

def t2e(T):
    return np.vstack([T[:,e] for e in (itertools.combinations(range(T.shape[1]),2))])


def msubplot(v_list,f_list,shape ,**sh):
    plt = None
    for i,(v,f) in enumerate(zip(v_list,f_list)):
        vw = mp.Viewer(sh)
        vw.add_mesh(v,f,shading=sh)
        plt = mp.Subplot(plt, vw, s=[shape[0],shape[1],i])
    return plt

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


def sync_camera(plt, plt0):
    '''empirical snippet to assign camera of plt0 to plt'''
    plt._cam.position = plt0._cam.position
    plt._orbit.exec_three_obj_method('update')
    plt._cam.exec_three_obj_method('updateProjectionMatrix')
    
            
def shrink(tetV,tetT, alpha):
    VT = tetV[tetT]
    mean = VT.mean(axis=1,keepdims=True)
    return (VT - mean)*alpha + mean
    

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