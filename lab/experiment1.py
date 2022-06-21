import torch
import arcsim
import sys
import json

sys.path.append('../pysim')

def reset_sim():
	arcsim.init_physics('curconf.json','',False)
	g = sim.gravity
	g.requires_grad = True
	return g

def run_sim():
	for step in range(49):
		arcsim.sim_step()

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)



def to_mesh(vertices, faces):
	pass


with open("./real/S1/sim_conf.json") as file:
    config = json.load(file)
save_config(config, 'curconf.json')


torch.set_num_threads(8)
sim=arcsim.get_sim()

g = reset_sim()
#run_sim()

#mat = sim.cloths[0].materials[0]

#print(mat)