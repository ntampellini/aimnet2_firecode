# coding=utf-8
'''
aimnet2-firecode
Copyright (C) 2024 Nicolò Tampellini

SPDX-License-Identifier: LGPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see
https://www.gnu.org/licenses/lgpl-3.0.en.html#license-text.

'''

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import torch
import io
from ase import Atoms
from ase.optimize import FIRE, LBFGS
from firecode.algebra import norm_of
from firecode.ase_manipulations import Spring
from firecode.calculators.__init__ import NewFolderContext
from firecode.calculators._xtb import xtb_gsolv
from firecode.optimization_methods import scramble_check
from firecode.utils import (align_structures, loadbar, time_to_string,
                            timing_wrapper, write_xyz)

from aimnet2_firecode.ase_calculator import AIMNet2Calculator


def get_aimnet2_model(method='wB97M-D3', gpu_rank=None, logfunction=print):
    '''
    method: string indicating the level of theory used to train the model
    rank: id of GPU to load the model onto. Ignored if model is loaded on CPU
    logfunction: function used to print execution messages
    '''

    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

    model_filename = {
                        'wb97m-d3' : f'{model_folder}/aimnet2_wb97m-d3_ens.jpt',
                        'b97-3c' : f'{model_folder}/aimnet2_b973c_ens.jpt'
                    }
    
    assert method.lower() in model_filename.keys(), f"method ({method.lower()}) not in {list(model_filename.keys())}"

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if torch.cuda.is_available():
        device = torch.device('cuda')

        if gpu_rank is not None:
            device = f'{device}:{gpu_rank}'

        if logfunction is not None:
            logfunction(f'--> {torch.cuda.device_count()} CUDA devices detected: running on GPU')

    else:
        device = torch.device('cpu')
        if logfunction is not None:
            logfunction('--> No CUDA devices detected: running on CPU')

    if logfunction is not None:
            logfunction('--> Loading AIMNet2 model from file', model_filename[method.lower()])
    
    model = torch.jit.load(model_filename[method.lower()], map_location=device)

    return model
    
def get_aimnet2_calc(method='wB97M-D3', gpu_rank=None, logfunction=print):
    return AIMNet2Calculator(get_aimnet2_model(method=method, gpu_rank=gpu_rank, logfunction=logfunction))

def get_shared_memory_aimnet2_model(method='wB97M-D3'):
    model = get_aimnet2_model(method=method, logfunction=None)
    buffer = io.BytesIO()
    torch.jit.save(model, buffer)
    model_bytes = buffer.getvalue()
    shared_array = torch.multiprocessing.Array('c', len(model_bytes))
    shared_array.raw = model_bytes
    return shared_array

def aimnet2_opt(
                coords,
                atomnos,
                ase_calc=None,
                method='wB97M-D3',
                constrained_indices=None,
                constrained_distances=None,
                charge=0,
                solvent=None,
                maxiter=500,
                conv_thr='tight',
                traj=None,
                logfunction=None,
                title='temp',

                optimizer='LBFGS',
                debug=False,
                **kwargs,
            ):
    '''
    embedder: firecode embedder object
    coords: 
    atomnos: 
    constrained_indices:
    safe: if True, adds a potential that prevents atoms from scrambling
    safe_mask: bool array, with False for atoms to be excluded when calculating bonds to preserve
    traj: if set to a string, traj is used as a filename for the bending trajectory.
    not only the atoms will be printed, but also all the orbitals and the active pivot.
    '''

    # create working folder and cd into it
    with NewFolderContext(title, delete_after=(not debug)):

        ase_calc = ase_calc or get_aimnet2_calc(method)
        ase_calc.do_reset()
        ase_calc.set_charge(charge)

        atoms = Atoms(atomnos, positions=coords)
        atoms.calc = ase_calc
        constraints = []

        if constrained_indices is not None:
            for i, c in enumerate(constrained_indices):
                i1, i2 = c
                tgt_dist = norm_of(coords[i1]-coords[i2]) if constrained_distances is None else constrained_distances[i]
                constraints.append(Spring(i1, i2, tgt_dist))

        atoms.set_constraint(constraints)

        fmax = {
            'tight' : 0.05,
            'loose' : 0.1,
        }[conv_thr]

        t_start_opt = time.perf_counter()
        optimizer_class = {'LBFGS':LBFGS, 'FIRE':FIRE}[optimizer]

        try:
            with optimizer_class(atoms, maxstep=0.1, logfile=None, trajectory=None if traj is None else 'temp.traj') as opt:
                opt.run(fmax=fmax, steps=maxiter)
                iterations = opt.nsteps

        except KeyboardInterrupt:
            print('KeyboardInterrupt requested by user. Quitting.')
            sys.exit()

        except TypeError:
            if logfunction is not None:
                logfunction(f'{title} in aimnet2_opt CRASHED')
            return coords, None, False 

        new_structure = atoms.get_positions()
        success = (iterations < 499)

        if logfunction is not None:
            exit_str = 'REFINED' if success else 'MAX ITER'
            logfunction(f'    - {title} {exit_str} ({iterations} iterations, {time_to_string(time.perf_counter()-t_start_opt)})')

        energy = atoms.get_total_energy() * 23.06054194532933 #eV to kcal/mol

        if traj is not None:
            os.system(f"ase convert temp.traj {traj}.xyz")

        try:
            os.remove('temp.traj')
            
        except FileNotFoundError:
            pass

        if solvent is not None:
            gsolv = xtb_gsolv(
                                new_structure,
                                atomnos,
                                model='alpb',
                                charge=charge,
                                solvent=solvent,
                                title=title,
                                assert_convergence=True,
                            )
            energy += gsolv

    return new_structure, energy, success

def aimnet2_optimization_refining(embedder, maxiter=None, conv_thr='tight', only_fixed_constraints=False):
    '''
    Refines structures by constrained optimizations with the active calculator,
    discarding similar ones and scrambled ones.
    maxiter - int, number of max iterations for the optimization
    conv_thr: convergence threshold, passed to calculator
    only_fixed_constraints: only uses fixed (UPPERCASE) constraints in optimization

    '''
    # run the serialized version if we only have one gpu,
    # as it avoids reloading the model for every structure
    # and it is actually faster
    if embedder.avail_gpus == 1:
        return aimnet2_optimization_refining_serial(embedder=embedder, maxiter=maxiter, conv_thr=conv_thr, only_fixed_constraints=only_fixed_constraints)

    embedder.outname = f'firecode_{"ensemble" if embedder.embed == "refine" else "poses"}_{embedder.stamp}.xyz'

    if only_fixed_constraints:
        task = 'Structure optimization (tight) / relaxing interactions'
    else:
        task = 'Structure optimization (loose)'

    # determining the device the model will run on
    device = 'gpu' if  torch.cuda.is_available() else 'cpu'

    # 1 GPU per worker if running on GPU, otherwise 4 CPUs per worker
    max_workers = torch.cuda.device_count() if device == 'gpu' else embedder.avail_cpus // 4

    solvent_line = f" + ΔGsolv[ALPB/{embedder.options.solvent}]" if embedder.options.solvent is not None else ""
    embedder.log(f'--> {task} ({embedder.options.theory_level}/vacuum{solvent_line}) level via {embedder.options.calculator}, {max_workers} thread{"s" if max_workers>1 else ""})')

    embedder.energies.fill(0)
    # Resetting all energies since we changed theory level

    t_start = time.perf_counter()
    processes = []
    cum_time = 0

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=torch.multiprocessing.get_context("spawn")) as executor:

        worker_func = aimnet2_worker

        for i, structure in enumerate(deepcopy(embedder.structures)):
            loadbar(i, len(embedder.structures), prefix=f'Optimizing structure {i+1}/{len(embedder.structures)} ')

            if only_fixed_constraints:
                constraints = np.array([value for key, value in embedder.pairings_table.items() if key.isupper()])
            
            else:
                constraints = np.concatenate([embedder.constrained_indices[i], embedder.internal_constraints]) if len(embedder.internal_constraints) > 0 else embedder.constrained_indices[i]

            pairing_dists = [embedder.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]

            gpu_rank = i % max_workers if device == 'gpu' else None

            process = executor.submit(
                                            timing_wrapper,
                                            worker_func,

                                            structure,
                                            embedder.atomnos,
                                            solvent=embedder.options.solvent,
                                            charge=embedder.options.charge,
                                            maxiter=maxiter,
                                            conv_thr=conv_thr,
                                            constrained_indices=constraints,
                                            constrained_distances=pairing_dists,
                                            title=f'Candidate_{i+1}',
                                            debug=embedder.options.debug,

                                            traj=f'Candidate_{i+1}_traj',
                                            # logfunction=embedder.log,
                                            
                                            cuda=(device=='gpu'),
                                            gpu_rank=gpu_rank,

                                            payload=(
                                                embedder.constrained_indices[i],
                                                )
                                        )
            
            processes.append(process)

        for i, process in enumerate(as_completed(processes)):
                    
            loadbar(i, len(embedder.structures), prefix=f'Optimizing structure {i+1}/{len(embedder.structures)} ')

            (   (
                new_structure,
                new_energy,
                embedder.exit_status[i]
                ),
            # from optimization function
                
                (
                embedder.constrained_indices[i],
                ),
            # from payload
            
                t_struct
            # from timing_wrapper

            ) = process.result()

            # assert that the structure did not scramble during optimization
            if embedder.exit_status[i]:
                constraints = (np.concatenate([embedder.constrained_indices[i], embedder.internal_constraints])
                                if len(embedder.internal_constraints) > 0
                                else embedder.constrained_indices[i])
                
                embedder.exit_status[i] = scramble_check(new_structure,
                                                    embedder.atomnos,
                                                    excluded_atoms=constraints.ravel(),
                                                    mols_graphs=embedder.graphs,
                                                    max_newbonds=0)
                
            cum_time += t_struct

            if embedder.options.debug:
                exit_status = 'REFINED  ' if embedder.exit_status[i] else 'SCRAMBLED'
                embedder.debuglog(f'DEBUG: optimzation_refining ({conv_thr}) - Candidate_{i+1} - {exit_status if new_energy is not None else "CRASHED"} {time_to_string(t_struct, digits=3)}')
            
            if embedder.exit_status[i] and new_energy is not None:
                embedder.structures[i] = new_structure
                embedder.energies[i] = new_energy

            else:
                embedder.energies[i] = 1E10

            ### Update checkpoint every (20*max_workers) optimized structures, and give an estimate of the remaining time
            chk_freq = int(embedder.avail_cpus//4) * embedder.options.checkpoint_frequency
            if i % chk_freq == chk_freq-1:

                with open(embedder.outname, 'w') as f:        
                    for j, (structure, status, energy) in enumerate(zip(align_structures(embedder.structures),
                                                                        embedder.exit_status,
                                                                        embedder.rel_energies())):

                        kind = 'REFINED - ' if status else 'NOT REFINED - '
                        write_xyz(structure, embedder.atomnos, f, title=f'Structure {j+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({embedder.options.ff_level})')

                elapsed = time.perf_counter() - t_start
                average = (elapsed)/(i+1)
                time_left = time_to_string((average) * (len(embedder.structures)-i-1))
                speedup = cum_time/elapsed
                embedder.log(f'    - Optimized {i+1:>4}/{len(embedder.structures):>4} structures - updated checkpoint file (avg. {time_to_string(average)}/struc, {round(speedup, 1)}x speedup, est. {time_left} left)', p=False)

    loadbar(1, 1, prefix=f'Optimizing structure {len(embedder.structures)}/{len(embedder.structures)} ')
    
    elapsed = time.perf_counter() - t_start
    average = (elapsed)/(len(embedder.structures))
    speedup = cum_time/elapsed

    embedder.log((f'{embedder.options.calculator}/{embedder.options.theory_level} optimization took '
            f'{time_to_string(elapsed)} (~{time_to_string(average)} per structure, {round(speedup, 1)}x speedup)'))

    ################################################# EXIT STATUS

    embedder.log(f'Successfully optimized {len([b for b in embedder.exit_status if b])}/{len(embedder.structures)} structures. Non-optimized ones will {"not " if not embedder.options.only_refined else ""}be discarded.')

    if embedder.options.only_refined:

        mask = embedder.exit_status
        embedder.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

        if False in mask:
            embedder.log(f'Discarded {len([b for b in mask if not b])} candidates for unsuccessful optimization ({np.count_nonzero(mask)} left')

    ################################################# PRUNING: ENERGY

    _, sequence = zip(*sorted(zip(embedder.energies, range(len(embedder.energies))), key=lambda x: x[0]))
    embedder.energies = embedder.scramble(embedder.energies, sequence)
    embedder.structures = embedder.scramble(embedder.structures, sequence)
    embedder.constrained_indices = embedder.scramble(embedder.constrained_indices, sequence)
    # sorting structures based on energy

    if embedder.options.debug:
        embedder.dump_status(f'optimization_refining_{conv_thr}', only_fixed_constraints=only_fixed_constraints)
        embedder.debuglog(f'DEBUG: Dumped emebedder status after optimizing candidates (\"optimization_refining_{conv_thr}\")')

    if embedder.options.kcal_thresh is not None and only_fixed_constraints:

        # mask = embedder.rel_energies() < embedder.options.kcal_thresh
        energy_thr = embedder.dynamic_energy_thr()
        mask = embedder.rel_energies() < energy_thr

        embedder.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

        if False in mask:
            embedder.log(f'Discarded {len([b for b in mask if not b])} candidates for energy ({np.count_nonzero(mask)} left, ' +
                        f'{round(100*np.count_nonzero(mask)/len(mask), 1)}% kept, threshold {energy_thr} kcal/mol)')

    ################################################# PRUNING: FITNESS (POST SEMIEMPIRICAL OPT)

    embedder.fitness_refining(threshold=2)

    ################################################# PRUNING: SIMILARITY (POST SEMIEMPIRICAL OPT)

    embedder.zero_candidates_check()
    embedder.similarity_refining()

    ################################################# CHECKPOINT AFTER SE OPTIMIZATION      

    with open(embedder.outname, 'w') as f:        
        for i, (structure, status, energy) in enumerate(zip(align_structures(embedder.structures),
                                                            embedder.exit_status,
                                                            embedder.rel_energies())):

            kind = 'REFINED - ' if status else 'NOT REFINED - '
            write_xyz(structure, embedder.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({embedder.options.ff_level})')

    embedder.log(f'--> Wrote {len(embedder.structures)} optimized structures to {embedder.outname}')

    # do not retain energies for the next optimization step if optimization was not tight
    if not only_fixed_constraints:
        embedder.energies.fill(0)

def aimnet2_worker(           
                    coords,
                    atomnos,
                    constrained_indices=None,
                    constrained_distances=None,
                    charge=0,
                    solvent=None,
                    maxiter=500,
                    conv_thr='tight',
                    traj=None,
                    logfunction=None,
                    title='temp',
                    debug=False,

                    cuda=False,
                    gpu_rank=None,
                ):
    
    ase_calc = get_aimnet2_calc(logfunction=None, gpu_rank=gpu_rank)
    
    opt_coords, energy, success = aimnet2_opt(
                                                coords,
                                                atomnos,
                                                ase_calc=ase_calc,
                                                constrained_indices=constrained_indices,
                                                constrained_distances=constrained_distances,
                                                charge=charge,
                                                solvent=solvent,
                                                maxiter=maxiter,
                                                conv_thr=conv_thr,
                                                traj=traj,
                                                logfunction=logfunction,
                                                title=title,
                                                debug=debug,
                                            )

    return opt_coords, energy, success

def aimnet2_optimization_refining_serial(embedder, maxiter=None, conv_thr='tight', only_fixed_constraints=False):
    '''
    Refines structures by constrained optimizations with the active calculator,
    discarding similar ones and scrambled ones.
    maxiter - int, number of max iterations for the optimization
    conv_thr: convergence threshold, passed to calculator
    only_fixed_constraints: only uses fixed (UPPERCASE) constraints in optimization

    '''

    embedder.outname = f'firecode_{"ensemble" if embedder.embed == "refine" else "poses"}_{embedder.stamp}.xyz'

    if only_fixed_constraints:
        task = 'Structure optimization (tight) / relaxing interactions'
    else:
        task = 'Structure optimization (loose)'

    solvent_line = f" + ΔGsolv[ALPB/{embedder.options.solvent}]" if embedder.options.solvent is not None else ""
    embedder.log(f'--> {task} ({embedder.options.theory_level}/vacuum{solvent_line}) level via {embedder.options.calculator}, single thread')

    embedder.energies.fill(0)
    # Resetting all energies since we changed theory level

    t_start = time.perf_counter()
    cum_time = 0

    for i, structure in enumerate(deepcopy(embedder.structures)):
        loadbar(i, len(embedder.structures), prefix=f'Optimizing structure {i+1}/{len(embedder.structures)} ')

        if only_fixed_constraints:
            constraints = np.array([value for key, value in embedder.pairings_table.items() if key.isupper()])
        
        else:
            constraints = np.concatenate([embedder.constrained_indices[i], embedder.internal_constraints]) if len(embedder.internal_constraints) > 0 else embedder.constrained_indices[i]

        pairing_dists = [embedder.get_pairing_dists_from_constrained_indices(_c) for _c in constraints]

        result = timing_wrapper(
                                        aimnet2_opt,

                                        structure,
                                        embedder.atomnos,
                                        ase_calc=embedder.dispatcher.aimnet2_model,
                                        solvent=embedder.options.solvent,
                                        charge=embedder.options.charge,
                                        maxiter=maxiter,
                                        conv_thr=conv_thr,
                                        constrained_indices=constraints,
                                        constrained_distances=pairing_dists,
                                        title=f'Candidate_{i+1}',
                                        debug=embedder.options.debug,

                                        traj=None,
                                        logfunction=embedder.log,

                                        payload=(
                                            embedder.constrained_indices[i],
                                            )
                                    )
                       
        loadbar(i, len(embedder.structures), prefix=f'Optimizing structure {i+1}/{len(embedder.structures)} ')

        (   (
            new_structure,
            new_energy,
            embedder.exit_status[i]
            ),
        # from optimization function
            
            (
            embedder.constrained_indices[i],
            ),
        # from payload
        
            t_struct
        # from timing_wrapper

        ) = result

        # assert that the structure did not scramble during optimization
        if embedder.exit_status[i]:
            constraints = (np.concatenate([embedder.constrained_indices[i], embedder.internal_constraints])
                            if len(embedder.internal_constraints) > 0
                            else embedder.constrained_indices[i])
            
            embedder.exit_status[i] = scramble_check(new_structure,
                                                embedder.atomnos,
                                                excluded_atoms=constraints.ravel(),
                                                mols_graphs=embedder.graphs,
                                                max_newbonds=0)
            
        cum_time += t_struct

        if embedder.options.debug:
            exit_status = 'REFINED  ' if embedder.exit_status[i] else 'SCRAMBLED'
            embedder.debuglog(f'DEBUG: optimzation_refining ({conv_thr}) - Candidate_{i+1} - {exit_status if new_energy is not None else "CRASHED"} {time_to_string(t_struct, digits=3)}')
        
        if embedder.exit_status[i] and new_energy is not None:
            embedder.structures[i] = new_structure
            embedder.energies[i] = new_energy

        else:
            embedder.energies[i] = 1E10

        ### Update checkpoint every (20*max_workers) optimized structures, and give an estimate of the remaining time
        chk_freq = int(embedder.avail_cpus//4) * embedder.options.checkpoint_frequency
        if i % chk_freq == chk_freq-1:

            with open(embedder.outname, 'w') as f:        
                for j, (structure, status, energy) in enumerate(zip(align_structures(embedder.structures),
                                                                    embedder.exit_status,
                                                                    embedder.rel_energies())):

                    kind = 'REFINED - ' if status else 'NOT REFINED - '
                    write_xyz(structure, embedder.atomnos, f, title=f'Structure {j+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({embedder.options.ff_level})')

            elapsed = time.perf_counter() - t_start
            average = (elapsed)/(i+1)
            time_left = time_to_string((average) * (len(embedder.structures)-i-1))
            speedup = cum_time/elapsed
            embedder.log(f'    - Optimized {i+1:>4}/{len(embedder.structures):>4} structures - updated checkpoint file (avg. {time_to_string(average)}/struc, {round(speedup, 1)}x speedup, est. {time_left} left)', p=False)

    loadbar(1, 1, prefix=f'Optimizing structure {len(embedder.structures)}/{len(embedder.structures)} ')
    
    elapsed = time.perf_counter() - t_start
    average = (elapsed)/(len(embedder.structures))
    speedup = cum_time/elapsed

    embedder.log((f'{embedder.options.calculator}/{embedder.options.theory_level} optimization took '
            f'{time_to_string(elapsed)} (~{time_to_string(average)} per structure, {round(speedup, 1)}x speedup)'))

    ################################################# EXIT STATUS

    embedder.log(f'Successfully optimized {len([b for b in embedder.exit_status if b])}/{len(embedder.structures)} structures. Non-optimized ones will {"not " if not embedder.options.only_refined else ""}be discarded.')

    if embedder.options.only_refined:

        mask = embedder.exit_status
        embedder.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

        if False in mask:
            embedder.log(f'Discarded {len([b for b in mask if not b])} candidates for unsuccessful optimization ({np.count_nonzero(mask)} left')

    ################################################# PRUNING: ENERGY

    _, sequence = zip(*sorted(zip(embedder.energies, range(len(embedder.energies))), key=lambda x: x[0]))
    embedder.energies = embedder.scramble(embedder.energies, sequence)
    embedder.structures = embedder.scramble(embedder.structures, sequence)
    embedder.constrained_indices = embedder.scramble(embedder.constrained_indices, sequence)
    # sorting structures based on energy

    if embedder.options.debug:
        embedder.dump_status(f'optimization_refining_{conv_thr}', only_fixed_constraints=only_fixed_constraints)
        embedder.debuglog(f'DEBUG: Dumped emebedder status after generating candidates (\"optimization_refining_{conv_thr}\")')

    if embedder.options.kcal_thresh is not None and only_fixed_constraints:

        # mask = embedder.rel_energies() < embedder.options.kcal_thresh
        energy_thr = embedder.dynamic_energy_thr()
        mask = embedder.rel_energies() < energy_thr

        embedder.apply_mask(('structures', 'constrained_indices', 'energies', 'exit_status'), mask)

        if False in mask:
            embedder.log(f'Discarded {len([b for b in mask if not b])} candidates for energy ({np.count_nonzero(mask)} left, ' +
                        f'{round(100*np.count_nonzero(mask)/len(mask), 1)}% kept, threshold {energy_thr} kcal/mol)')

    ################################################# PRUNING: FITNESS (POST SEMIEMPIRICAL OPT)

    embedder.fitness_refining(threshold=2)

    ################################################# PRUNING: SIMILARITY (POST SEMIEMPIRICAL OPT)

    embedder.zero_candidates_check()
    embedder.similarity_refining()

    ################################################# CHECKPOINT AFTER SE OPTIMIZATION      

    with open(embedder.outname, 'w') as f:        
        for i, (structure, status, energy) in enumerate(zip(align_structures(embedder.structures),
                                                            embedder.exit_status,
                                                            embedder.rel_energies())):

            kind = 'REFINED - ' if status else 'NOT REFINED - '
            write_xyz(structure, embedder.atomnos, f, title=f'Structure {i+1} - {kind}Rel. E. = {round(energy, 3)} kcal/mol ({embedder.options.ff_level})')

    embedder.log(f'--> Wrote {len(embedder.structures)} optimized structures to {embedder.outname}')

    # do not retain energies for the next optimization step if optimization was not tight
    if not only_fixed_constraints:
        embedder.energies.fill(0)