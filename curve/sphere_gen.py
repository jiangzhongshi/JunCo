"""
Some script to generate an example high order sphere.
"""
if __name__ == "__main__":
  import fem_tabulator
else:
  from . import fem_tabulator
import meshzoo
import meshio
import numpy as np


def codec_to_n(co): return [k for i, j in enumerate(co) for k in [i]*j]

def sphere_gen(order, ico):
    V, F = meshzoo.icosa_sphere(ico)
    if order == 1:
      return meshio.Mesh(points = V, cells = [('triangle',F)])
    el = (order+1)*(order+2)//2

    tri10 = fem_tabulator.basis_info(order=order, nsd=2, force_codec=f'tri{el}')
    pts_o3 = np.array(tri10['codec'])/order

    cod_n = np.array([codec_to_n(c) for c in tri10['codec']])

    lin_cp = np.einsum('fed,Ee->fEd', V[F], pts_o3)

    sphere_cp = 50*(lin_cp/np.linalg.norm(lin_cp, axis=-1, keepdims=True))

    comb_tuples = F[:, cod_n].reshape(-1, order)

    uniq_tuples, uniq_ind, uniq_inv = np.unique(comb_tuples,
                                                axis=0, return_index=True, return_inverse=True)

    cell_str = f'triangle{el}'
    m = meshio.Mesh(points=sphere_cp.reshape(-1, 3)[uniq_ind],
                    cells=[(cell_str, uniq_inv[np.arange(len(sphere_cp)*el).reshape(-1, el)])])
    return m


# for ico,name in zip([50, 10, 5],[1,5,10]):
#   for order in [2,3]:
#     m = sphere_gen(order, ico)
#     meshio.write(f"Sphere{name}WL_o{order}.msh", m, binary=False)
import tqdm
for ico in tqdm.trange(1,101):
  for order in [1, 2,3]:
    m = sphere_gen(order, ico)
    meshio.write(f'./spherewl/Sphere_icosa{ico}_o{order}.msh', m, binary=False)
  