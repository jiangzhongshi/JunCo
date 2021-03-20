from vis_utils import *

refV,refF,sB,sT,sV,sF,trackers = h5deser('/home/zhongshi/Workspace/nutshell_data/raw_data/leg-intersect.off.h5')

import sys
sys.path.append('../python')
import seism

stdV,stdF = igl.upsample(np.array([[0,0],[1,0],[0,1.]]),np.array([[0,1,2]]),2)

u,v = stdV[:,:1], stdV[:,1:]
allV = np.einsum('fjk,bj->fbk', sV[sF], np.hstack([1-u-v, u, v]))
allF = np.vstack([stdF+len(u)*i for i in range(len(sF))])

allB = np.einsum('fjk,bj->fbk', sB[sF], np.hstack([1-u-v, u, v])).reshape(-1,3)
allT = np.einsum('fjk,bj->fbk', sT[sF], np.hstack([1-u-v, u, v])).reshape(-1,3)


secV, SVI, SVJ, _ = igl.remove_duplicate_vertices(allV.reshape(-1,3),allF, 1e-10)
secF = SVJ[allF]
# c=(np.arange(len(secF))//16)
# secV == SVI[allV]

secB, secT = allB[SVI], allT[SVI]

v_parent_f = np.repeat(np.arange(len(sF)),16)[SVI]

import sys

results = -np.ones((len(secB), 4))
for fi in range(20):
    print(fi)
    sys.stdout.flush()
    track_f = trackers[fi]
    if len(track_f) == 1:
      track_f = [track_f]
    lV, lF, _, _ = igl.remove_unreferenced(refV,refF[track_f])
    if len(track_f) == 1:
      lF = np.array([lF])
    assert len(lF) == len(trackers[fi])
    tree = seism.AABB(lV,lF)
    v_to_consider = np.where(v_parent_f==fi)[0]
    for v in v_to_consider:
        print(v)
        sys.stdout.flush()

        hf, hu, hv, ht = (tree.segment_hit(secB[v], secT[v], False))
        results[v] = np.array([track_f[hf], 1-hu-hv, hu, hv])