import ase
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.io import read,write
from ase.visualize import view
from ase.build import minimize_rotation_and_translation
import matplotlib.pyplot as plt
import numpy as np
import schnetpack
from schnetpack.data import ASEAtomsData

md_trajectory = read('ethanol_data.xyz', index=':')
md_trajectory2 = read('moldyn_400.traj', index=':')
# print(md_trajectory[0]) 
# print(md_trajectory2[0]) 
# view(md_trajectory2)

e_a = md_trajectory2[0].get_potential_energy()

energies = [md_trajectory2[i].get_potential_energy() for i in range(len(md_trajectory2))]

plt.figure()
plt.plot(energies)
plt.ylabel("Potential Energy", fontsize=16)
plt.xlabel("Saved Step (every 50 steps)", fontsize=16)
plt.tight_layout()
#plt.show()
plt.savefig("energy_our_model3.png")

#md_trajectory = a

min_a = md_trajectory2[0]

for i in range(len(md_trajectory2)):
    minimize_rotation_and_translation(min_a, md_trajectory2[i])

def rmsd(m1, m2):
    r1 = m1.get_positions()
    r2 = m2.get_positions()
    return np.sqrt(np.mean((r1-r2)**2))

rmsd_values = [rmsd(molecule, min_a) for molecule in md_trajectory2]

plt.figure()
plt.plot(rmsd_values)
plt.ylabel("RMSD", fontsize=16)
plt.xlabel("Saved Step (every 50 steps)", fontsize=16)
#plt.show()
plt.savefig("rmsd_our_model3.png")

local_min_idx = np.argmin(energies[1:])
local_min_molecule = md_trajectory2[local_min_idx]
print(local_min_idx)
#view(local_min_molecule)

#view(a[2745])
    

# db_path = "ethanol_data.db"

# properties = [
#     dict(
#         energy=np.array([molecule.get_potential_energy()]),
#         forces=np.array(molecule.get_forces()),
#     )
#     for molecule in md_trajectory
# ]

# dataset = ASEAtomsData.create(
#     datapath=db_path,
#     distance_unit = "Ang",
#     property_unit_dict={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
# )

# dataset.add_systems(
#     atoms_list = md_trajectory,
#     property_list=properties,
# )

# print(properties[0] ['energy'])
# print(properties[0]['forces'])
