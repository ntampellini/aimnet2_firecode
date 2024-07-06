import sys

import torch
from firecode.utils import read_xyz, write_xyz

from aimnet2_firecode.interface import aimnet2_opt, get_aimnet2_calc

if __name__ == '__main__':

    energies = []
    calc = get_aimnet2_calc()

    for filename in sys.argv[1:]:

        mol = read_xyz(filename)
        coords = mol.atomcoords[0]
        basename = filename.split(".")[0]

        opt_coords, energy, success = aimnet2_opt(
                                                    torch.array(coords, dtype=torch.float32),
                                                    mol.atomnos,
                                                    calc=calc,
                                                    # traj=basename+"_trj.xyz",
                                                    logfunction=print,
                                                )
        
        with open(basename+"_opt.xyz", 'w') as f:
            write_xyz(opt_coords, mol.atomnos, f)

        energies.append(energy)

    min_e = min(energies)

    for filename, energy in zip(sys.argv[1:], energies):
        rel_energy = energy - min_e
        print(f"{filename:<20} : {rel_energy:.2f} kcal/mol")