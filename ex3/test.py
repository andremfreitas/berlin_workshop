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

a = read('ethanol_data.xyz', index=':')
#print(len(a)) 

#view(a, viewer = 'x3d')

#e_a = a[0].get_potential_energy()

#energies = [a[i].get_potential_energy() for i in range(len(a))]

# plt.figure()
# plt.plot(energies)
# plt.ylabel("Energy")
# plt.xlabel("Step")
# plt.tight_layout()
# plt.show()
md_trajectory = a
# min_a = a[0]

# for i in range(len(a)):
#     minimize_rotation_and_translation(min_a, a[i])

# def rmsd(m1, m2):
#     r1 = m1.get_positions()
#     r2 = m2.get_positions()
#     return np.sqrt(np.mean((r1-r2)**2))

# rmsd_values = [rmsd(molecule, min_a) for molecule in a]

# plt.figure()
# plt.plot(rmsd_values)
# plt.show()


#view(a[2745])
    

db_path = "ethanol_data.db"

properties = [
    dict(
        energy=molecule.get_potential_energy(),
        forces=molecule.get_forces(),
    )
    for molecule in md_trajectory
]

dataset = ASEAtomsData.create(
    datapath=db_path,
    distance_unit = "Ang",
    property_unit_dict=dict(energy="kcal/mol", forces="kcsl/mol/Ang"),
)

dataset.add_systems(
    atoms_list = md_trajectory,
    property_list=properties,
)
