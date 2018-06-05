# GROMACS
Commands and scripts for MD using GROMACS

Status: Not complete. Currently porting a more complex module to a stripped-back simplified version.

Creates shell scripts for running full the MD process from start to finish with some simple analysis too. Only uses Python standard library, but GROMACS (gmx) will need to be installed locally if one wishes to run anything locally (this can be a pain in Windows). 

For HPC integration with BlueCrystal (UOB only), ssh-keygen and ssh-copy-id will need to be on the system path - if using Windows + Cygwin, these are in the C:/cygwin/bin directory. This is not fully implemented yet - the local script produced has be run manually.

## Usage

Before running, create a directroy to work in on the local side and place the pdb(s) and MDPs here. Edit the MDP files to your liking, e.g. increase nsteps in md.mdp for longer simulations. 

Simple Example:

    import gmx
    
    gmx.USER = 'yourname' # UOB BlueCrystal account username
    # BCP3 used currently. Set gmx.BCP3 = '@path.to.bcp4' for BCP4 submission (not currently configured).
    
    g = gmx.Gmx(
        'mol.pdb',
        cwd='path/to/work/directory',
        remote=True,                  # for remote runs only
        mpi=True,                     # set to True for remote runs, even if only using one node
        key_gen=False,                # creates RSA key pair for ssh if True
        hrs=24,                       # hours to request in qsub
        num_nodes=1,                  # number of nodes to run on
        gpu=1)                        # request a gpu - increases speed
        
    g.run_basic(
        all_remote=True,                            # runs all commands remotely
        trjconv=dict(center_group=1, out_group=1),  # trjconv options
        rmsd=dict(fit_group1=4, fit_group2=4))      # rms options
  
If running in `remote` or `dry` mode, only the run scripts are created and nothing is executed. If not `dry` or `remote` then it will attempt to run locally.
