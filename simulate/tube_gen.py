import numpy as np
import meshio
def tetmesh_from_shell(base, top, F):
    tetra_splits = (np.array([0, 3, 4, 5, 1, 4, 2, 0, 2, 5, 0, 4]).reshape(-1, 4),
                    np.array([0, 3, 4, 5, 1, 4, 5, 0, 2, 5, 0, 1]).reshape(-1, 4))
    vnum = len(base)
    T = []
    for f in F:
        tet_c = tetra_splits[0] if f[1] > f[2] else tetra_splits[1]
        T.append((tet_c // 3)*vnum + f[tet_c % 3])
    return np.vstack([base, top]), np.vstack(T)

def tube(length: float = 1.0, radius: float = 1.0, n: int = 30, nw : int = -1, distribute = None):
    """tube Function from nschole/MeshZoo"""
    # Number of nodes along the width of the strip (>= 2)
    # Choose it such that we have approximately square boxes.
    if nw == -1:
        nw = int(round(length * n / (2 * np.pi * radius)))

    # Generate suitable ranges for parametrization
    u_range = np.linspace(0.0, 2 * np.pi, num=n, endpoint=False)
    v_range = np.linspace(0 * length, 1 * length, num=nw)
    if distribute is not None:
        v_range = distribute(np.linspace(0, 1, num=nw))

    # Create the vertices.
    proto_nodes = np.dstack(np.meshgrid(u_range, v_range, indexing="ij")).reshape(-1, 2)
    nodes = np.column_stack(
        [
            proto_nodes[:, 1],
            radius * np.cos(proto_nodes[:, 0]),
            radius * np.sin(proto_nodes[:, 0]),
        ]
    )

    # create the elements (cells)
    elems = []
    for i in range(n - 1):
        for j in range(nw - 1):
            elems.append([i * nw + j, (i + 1) * nw + j + 1, i * nw + j + 1])
            elems.append([i * nw + j, (i + 1) * nw + j, (i + 1) * nw + j + 1])

    # close the geometry
    for j in range(nw - 1):
        elems.append([(n - 1) * nw + j, j + 1, (n - 1) * nw + j + 1])
        elems.append([(n - 1) * nw + j, j, j + 1])

    return nodes, np.array(elems)

def shell_gen(n, nw, adapt=False):
    func = None
    if adapt:
        func =  lambda x: 16 * np.concatenate([np.linspace(i/3, (i+1)/3, l,endpoint=(i==2)) for i,l in enumerate([5,len(x) - 10,5])])

    base, f_b = tube(length=16.0, radius=0.9, n=n, nw = nw, distribute=func)
    top, f_t = tube(length=16.0, radius=1.0, n=n, nw = nw, distribute=func)
    v, t = tetmesh_from_shell(base, top, f_b)
    v /= 16.0
    print(n,nw, t.shape)
    meshio.write(f'simulate/data/tube_{n}_{nw}_A{adapt}.msh', meshio.Mesh(points = v, cells = [('tetra', t)]))

if __name__ == '__main__':
    for n in [3,5,10, 20]:
        for nw in [10, 20,40,60,80]:
            shell_gen(n, nw)
