import tempfile
import numpy as np
import subprocess

def refine_tris(v,f, a,flag='-raYY'):
    nodelines = [f'{len(v)} 2 0 0'] + [f'{i} {vi[0]} {vi[1]}' for i, vi in enumerate(v, 1)]
    elelines = [f'{len(f)} 3 0'] + [f'{i} {fi[0]} {fi[1]} {fi[2]}' for i, fi in enumerate(f+1, 1)]
    arealines = [f'{len(a)}'] + [f'{i} {ai}' for i, ai  in enumerate(a, 1)]
    with tempfile.TemporaryDirectory() as dirname:
        name = dirname + '/tmptest'
        with open(f'{name}.node','w+') as fp:
            fp.write('\n'.join(nodelines))
        with open(f'{name}.ele','w+') as fp:
            fp.write('\n'.join(elelines))
        with open(f'{name}.area','w+') as fp:
            fp.write('\n'.join(arealines))
        info = (subprocess.run(f'/home/zhongshi/Workspace/triangle/triangle {flag} {name}', 
                               shell=True,capture_output=True))
        return read_node_ele(name + '.1')
rv ,rf = refine_tris(tv,tf,np.exp(-10*np.linalg.norm(bc,axis=1)-2), flag='-raq')

def read_node_ele(name):
    with open(f'{name}.node') as fp:
        nodelines = [l for l in fp.readlines()]
        nodelines = [list(map(float, l.split()[1:-1]))
                     for l in nodelines[1:] if l[0]!='#']

    with open(f'{name}.ele') as fp:
        elelines = [l for l in fp.readlines()]
        elelines = [list(map(int, l.split()[1:]))
                     for l in elelines[1:] if l[0]!='#']
    return np.array(nodelines), np.array(elelines)-1

def triangle_triangulate(v,e,flag='-p'):
    lines = ([f'{len(v)} 2 0 1'] + 
             [f'{i} {vv[0]} {vv[1]} 1' 
                       for i,vv in enumerate(v,1)] +
            [f'{len(e)} 1'] +
                     [f'{i} {ee[0]} {ee[1]} 1' for i, ee in enumerate(e+1, 1)] +
            ['0'] # no holes
            )
    with tempfile.TemporaryDirectory() as dirname:
        name = dirname + '/tmptest'
        with open(f'{name}.poly','w+') as fp:
            fp.write('\n'.join(lines))
        outputinfo = subprocess.run(f'/home/zhongshi/Workspace/triangle/triangle {flag} {name}.poly', shell=True, capture_output=True)
        nodes, eles =  read_node_ele(name + '.1.')
    return nodes, eles

area = 0.1
def triangular_tube(area, axial_length = 16, radius=1):
    edgelen = np.sqrt(area)
    axial_cnt = int(axial_length/edgelen)
    circu_cnt = int(2*np.pi*radius/edgelen)
    v = np.array([(x,0) for x in np.linspace(0,1,axial_cnt,endpoint=False)] + 
                 [(1,y) for y in np.linspace(0,1,circu_cnt,endpoint=False)] + 
                 [(x,1) for x in np.linspace(1,0,axial_cnt,endpoint=False)] + 
                 [(0,y) for y in np.linspace(1,0,circu_cnt,endpoint=False)])
    numv = len(v)
    e = np.array([(i,(i+1)%numv) for i in range(numv)])

    tv, tf = triangle_triangulate(v*[axial_length,2*np.pi*radius], e, f'-pYYPQa{area}')
    roll_v = np.stack([tv[:,0], np.sin(tv[:,1]), np.cos(tv[:,1])]).T

    roll_map = np.arange(len(tv))
    a_c = axial_cnt + circu_cnt
    for j,i in enumerate(range(a_c, a_c+axial_cnt + 1)):
        roll_map[i] = axial_cnt  - j
    roll_map[a_c+axial_cnt +1:] -= axial_cnt+1

    roll_ind = np.concatenate([np.arange(a_c),
                               np.arange(a_c+axial_cnt + 1, len(tv))])

    rv, rf = roll_v[roll_ind], roll_map[tf]
    def roll_face(f):
        return np.roll(f, -f.argmin())
    rf = np.array([roll_face(f) for f in rf])
    return rv, rf