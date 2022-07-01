from json import load
import torch
import arcsim

from pytorch3d.io import load_obj, save_obj, IO, load_objs_as_meshes

mesh = load_objs_as_meshes(["real/S1/templates/template_mesh_final.obj"])

IO().save_mesh(mesh, "cloth_mesh_0.obj")
