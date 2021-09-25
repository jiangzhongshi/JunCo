import numpy as np
import igl
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


def tube(length: float = 1.0, radius: float = 1.0, n: int = 30, nw: int = -1, distribute=None):
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
    proto_nodes = np.dstack(np.meshgrid(
        u_range, v_range, indexing="ij")).reshape(-1, 2)
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

    # Quads
    # for i in range(n - 1):
    #     for j in range(nw - 1):
    #         elems.append([i * nw + j, (i + 1) * nw + j, (i + 1) * nw + j + 1, i * nw + j + 1])

    # # close the geometry
    # for j in range(nw - 1):
    #     elems.append([(n - 1) * nw + j, j, j + 1, (n - 1) * nw + j + 1])

    return nodes, np.array(elems)


def shell_gen(n, nw, upsample=0, adapt=False):
    func = None
    if adapt:
        def func(x): return 16 * np.concatenate([np.linspace(
            i/3, (i+1)/3, l, endpoint=(i == 2)) for i, l in enumerate([5, len(x) - 10, 5])])
    verts, faces = tube(length=16.0, radius=1, n=n, nw=nw, distribute=func)
    for _ in range(upsample):
        verts, faces = igl.upsample(verts, faces)

    v, t = tetmesh_from_shell(verts*[1, .9, .9], verts, faces)
    v /= 16.0
    print(n, nw, t.shape)
    adapt_token = 'A' if adapt else 'U'
    meshio.write(f'data/tube_{n}_{nw}_{adapt_token}_up{upsample}.msh',
                 meshio.Mesh(points=v, cells=[('tetra', t)]))


if __name__ == '__main__':
    import fire
    fire.Fire(shell_gen)
    # for n in [30,40,50]:
    # for nw in [30, 40, 50]:
    # shell_gen(n, nw, upsample = 0)
