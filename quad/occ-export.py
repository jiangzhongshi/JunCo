from OCC.Core.TColgp import TColgp_HArray1OfPnt2d, TColgp_Array1OfPnt2d,TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt2d,gp_Vec, gp_Pnt
from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface
from OCC.Core.TColGeom import TColGeom_Array2OfBezierSurface
from OCC.Core.GeomConvert import GeomConvert_CompBezierSurfacesToBSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

def simple_vis(surf):
	from OCC.Display.SimpleGui import init_display
	display, start_display, add_menu, add_function_to_menu = init_display()
	display.EraseAll()
	display.DisplayShape(surf, update=True)
	start_display()

def cp_to_bz(cp):
    array1 = TColgp_Array2OfPnt(1, len(cp), 1, len(cp))
    for i in range(1,len(cp)+1):
        for j in range(1,len(cp)+1):
            array1.SetValue(i,j, gp_Pnt(*cp[i-1,j-1]))
    BZ1 = Geom_BezierSurface(array1)
    return BZ1

def write_to_step(output_file, quad_cp):
	from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
	from OCC.Core.Interface import Interface_Static_SetCVal
	from OCC.Core.IFSelect import IFSelect_RetDone
	from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
	step_writer = STEPControl_Writer()
	Interface_Static_SetCVal("write.step.schema", "AP203")

	for cp in quad_cp:
		assert len(cp) == 16
		b1 = cp_to_bz(cp.reshape(4,4,3))
		build = OCC.Core.BRepBuilderAPI.BRepBuilderAPI_MakeFace(b1, 1e-6)
		step_writer.Transfer(build.Shape(),STEPControl_AsIs)
	status = step_writer.Write(output_file)

def compose_bezier(BZ1):
	bezierarray = TColGeom_Array2OfBezierSurface(1, 1, 1, 1)
	bezierarray.SetValue(1, 1, BZ1)
	BB = GeomConvert_CompBezierSurfacesToBSplineSurface(bezierarray)
	if BB.IsDone():
		poles = BB.Poles().Array2()
		uknots = BB.UKnots().Array1()
		vknots = BB.VKnots().Array1()
		umult = BB.UMultiplicities().Array1()
		vmult = BB.VMultiplicities().Array1()
		udeg = BB.UDegree()
		vdeg = BB.VDegree()
		BSPLSURF = Geom_BSplineSurface( poles, uknots, vknots, umult, vmult, udeg, vdeg, False, False)

		return BSPLSURF
	else:
		return None
