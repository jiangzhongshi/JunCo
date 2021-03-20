import glob
import os
import sys
import pymesh

#refV,refF,B,T,V,F,_=h5parse(sys.argv[1])
mesh = pymesh.load_mesh(sys.argv[1])
tetgen = pymesh.tetgen()
tetgen.points = mesh.vertices
tetgen.triangles=mesh.faces
tetgen.verbosity=3
tetgen.coarsening=False
#tetgen.split_boundary=False

tetgen.run();
pymesh.save_mesh(sys.argv[2], tetgen.mesh)
