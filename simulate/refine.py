import numpy as np

def quad_to_tris(q):
    return np.vstack([q[:,[0,2,3]], q[:, [0,1,2]]])

def propogate_colors(tt, colors):
    color_sum = 0
    while color_sum != colors.sum():
        color_sum = colors.sum()
        for i in range(len(tt)):
            if colors[i] == 2: # red
                continue
            cnt = 0
            for j in range(3):
                if tt[i,j] != -1 and colors[tt[i,j]] == 2:
                    cnt += 1
            if cnt > 1:
                colors[i] = 2
            if cnt == 1:
                colors[i] = 1
    return colors
                
def subdivide_tris(verts, faces, marker):
    tt,tti = igl.triangle_triangle_adjacency(faces)
    edge_marker = -np.ones_like(tt)
    colors = marker*2
    colors = propogate_colors(tt, colors)
    indices = np.nonzero(colors==2)[0]
    
    newf = []
    newv = []
    nv = len(verts)
    for fi in indices:
        # split
        for j in range(3):
            if edge_marker[fi,j] == -1:
                edge_marker[fi, j] = nv
                fo, jo = tt[fi,j], tti[fi,j]
                if fo != -1:
                    edge_marker[fo,jo] = nv
                nv += 1
                newv.append((verts[faces[fi,j]] + verts[faces[fi,(j+1)%3]])/2)
        newf += [(faces[fi,j], edge_marker[fi,j], edge_marker[fi,(j+2)%3]) for j in range(3)] + [edge_marker[fi]]
    green_ids = np.nonzero(colors==1)[0]
    for fi in green_ids:
        idx = np.argmax(edge_marker[fi])
        val = edge_marker[fi,idx]
        assert val >= 0
        fr = np.roll(faces[fi], -idx)
        newf += [[fr[0], val, fr[2]], [fr[2], val, fr[1]]]
        
    return np.vstack([verts,newv]), np.vstack([faces[colors==0], newf])