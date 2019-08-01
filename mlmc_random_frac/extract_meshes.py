import os
import sys
from shutil import copyfile

if len(sys.argv) > 1:
    mesh_repo = sys.argv[1]
else:
    mesh_repo = "mesh_repository"
    
samples_dir = os.path.join("output_1", "sim_0_step_0.010000", "samples")
mesh_repository_dir = os.path.join(mesh_repo)
os.makedirs(mesh_repository_dir, exist_ok=True)

meshes = {}
mesh_indices = []
for mesh_file in os.listdir(os.fsencode(mesh_repository_dir)):
    mesh_path = os.path.join(mesh_repository_dir, os.fsdecode(mesh_file))
    info = os.stat(mesh_path)
    meshes[info.st_size] = mesh_path
    tags = os.fsdecode(mesh_file).split('_')
    mesh_indices.append(int(tags[0]))
mesh_indices.sort()
if mesh_indices:
  last_index = mesh_indices[-1]
else:
  last_index = 0

print(samples_dir)
for sdir in os.listdir(os.fsencode(samples_dir)):  
    #print(sdir)
    sdir = os.path.join(samples_dir,os.fsdecode(sdir))
    if os.path.isdir(sdir):
        finished_path = os.path.join(sdir, "FINISHED")
        #print(finished_path)
        if os.path.isfile(finished_path):
            #print(sdir, " FINISHED")
            mesh_file = "random_fractures_healed.msh"
            mesh_path = os.path.join(sdir, mesh_file )
            if os.path.isfile(mesh_path):             
                info = os.stat(mesh_path)
                if info.st_size not in meshes:
                    last_index += 1
                    copy_mesh_file = "{:04d}_{}".format(last_index, mesh_file)
                    copyfile(mesh_path, os.path.join(mesh_repository_dir, copy_mesh_file))
                    print("Copy {} to {}.".format(mesh_path, copy_mesh_file))
                else:
                    print("Skipping mesh {} same size as {}.".format(mesh_path, meshes[info.st_size]))

    
