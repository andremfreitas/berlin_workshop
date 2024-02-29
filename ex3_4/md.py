import torch
import schnetpack as spk
from ase.io import read
import numpy as np


model = torch.load("best_model", map_location=torch.device('cpu'))
cutoff_radius = model.representation.cutoff_fn.cutoff.item()

spk_calculator = spk.interfaces.SpkCalculator(
    model_file="best_model",
    neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff_radius),
)

at = read("./ethanol_data.xyz", index="0")
at.pbc = np.array([False, False, False])
at.calc = spk_calculator




"""Demonstrates molecular dynamics with constant temperature."""

# from asap3 import EMT  # Way too slow with ase.EMT !

from ase import units
from ase.io.trajectory import Trajectory    
from ase.md.langevin import Langevin

T = 400  # Kelvin

# Lukas -- 350, 2, 0.1
# Mine -- 500, 1, 0.002

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(at, timestep=2*units.fs, temperature_K = T, friction= 0.1 / units.fs)


def printenergy(a=at):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(
        "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
        "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
    )   


dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory("moldyn_400.traj", "w", at)
dyn.attach(traj.write, interval=50)

# Now run the dynamics
printenergy()
dyn.run(100000)
