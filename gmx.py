"""
Created on 12 Apr 2017

@author: mb1511
"""

from __future__ import print_function

import subprocess as sp
import glob
import os
import logging
import multiprocessing
from os.path import basename, join, abspath

# configuration 
GMX = 'gmx'
GMX_MPI = 'gmx_mpi'
MACHINE_FILE = 'machine.file.$PBS_JOBID'   # can be anything, but this will add the BC job id
NUM_PROCS = 'numnodes'
PDB2GMX = 'pdb2gmx'
EDITCONF = 'editconf'
SOLVATE = 'solvate'
GENION = 'genion'
GROMPP = 'grompp'
MDRUN = 'mdrun'
TPBCONV = 'convert-tpr'
TRJCONV = 'trjconv'
RMSD = 'rms'
USER = ''
BCP3 = 'bluecrystalp3.acrc.bris.ac.uk'
MPI_QSUB = '''#!/bin/bash
# request resources
#PBS -l nodes={nodes}:ppn={ppn}{gpu}
#PBS -l walltime={hrs:02d}:{mins:02d}:{sec:02d}

# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1

# define GROMACS version
module add apps/gromacs-5.0-gnu-mpi-gpu-plumed

# record some potentially useful details about the job:
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo PBS job ID is $PBS_JOBID
echo This jobs runs on the following nodes:
echo `cat $PBS_NODEFILE | uniq`

# setup mpi run config
cat $PBS_NODEFILE > {machine_file}
{num_procs}=`wc $PBS_NODEFILE | awk '{{print $1}}'`
'''


def runmethod(name=''):
    def run(f):
        def wrap(cls, **kw):
            dry = cls.dry
            print('========== ' + name + ' ==========\n')
            scp = False
            all_remote = kw.pop('all_remote', False)
            if all_remote:
                scp = True
            else:
                for k in kw:
                    if 'remote' in kw[k].keys():
                        scp = True
                        break
            
            print('Local Script:\n\n')
            cls.local_script += '#!/bin/bash\n'
            for cmd in f(cls):
                opts = kw.get(cmd.__name__, {})
                if all_remote:
                    opts.update(remote=True)
                if not dry:
                    rc = cmd(**opts)
                    if rc[1]:
                        cls.qsub += rc[0]
                    elif rc[0]:
                        print('Error while performing: {}'.format(cmd.__name__))
                        print('See log file(s) for details')
                        break
                else:
                    text = cmd(**opts)
                    if text[1] or all_remote:
                        cls.qsub += '\n' + text[0]
                    elif text[0]:
                        cls.local_script += text[0] + '\n'
                        
            if scp:
                cls.local_script += 'scp -r {ssh_key}../{cwd} {user}@{bcp}:~/{remote_dir}\n'.format(
                    cwd=basename(cls.cwd), user=USER, bcp=BCP3, remote_dir='md',
                    ssh_key='-i {} '.format(cls.ssh_key) if cls.ssh_key else '')
                
                cls.local_script += "ssh -X {ssh_key}{user}@{bcp} 'cd {remote_dir}/{cwd}; qsub {qsub}'\n".format(
                    user=USER, bcp=BCP3, remote_dir='md',
                    cwd=basename(cls.cwd), qsub='gmxqs',
                    ssh_key='-i {} '.format(cls.ssh_key) if cls.ssh_key else '')
                
            print(cls.local_script)
            print ('\n\nSubmission Script:\n')
            print(cls.qsub)
            cls.save_scripts()
        return wrap
    return run

def ssh_keygen(user, host, key_name='id_rsa', cwd='./', copy=True):
    cmd = [
        'ssh-keygen',
        '-f', key_name,
        '-N', "''"]
    print(' '.join(cmd))
    sp.call(cmd, cwd=cwd)
    if copy:
        cmd = """bash -c 'ssh-copy-id "$0" "$1" "$2"' -i {0} {1}@{2}""".format(
            key_name, user, host)
        sp.call(cmd, cwd=cwd)

class Gmx():
    """
    GROMACS container
    """

    def __init__(
            self, pdb1, pdb2='', cwd='./', dry=True, num_threads=6,
            remote=False, mpi=False, key_gen=False, ssh_key='id_rsa',
            num_nodes=4, ppn=16, hrs=72, mins=0, sec=0, gpu=0,
            loglevel='DEBUG', quiet=False):
        self._pdb1 = pdb1
        self._pdb2 = pdb2
        self.cwd = abspath(cwd)
        self.quiet = quiet
        self.dry = dry
        
        # loacal options
        self.num_threads = num_threads
        self.local_script = ''
        
        # remote options
        self.mpi = mpi
        self.num_nodes = num_nodes
        self.qsub = ''
        if gpu:
            gpu = ':gpus={}'.format(gpu)
        else:
            gpu = ''
        
        assert self.num_threads <= multiprocessing.cpu_count()
        if remote:
            assert USER, 'Please define gmx.USER for remote submissions.'
            
            if key_gen:
                ssh_keygen(USER, BCP3, key_name=ssh_key, cwd=self.cwd)
                self.ssh_key = ssh_key
            else:
                self.ssh_key = ssh_key
            
            self.qsub = MPI_QSUB.format(
                nodes=num_nodes, ppn=ppn,
                machine_file=MACHINE_FILE, num_procs=NUM_PROCS,
                hrs=hrs, mins=mins, sec=sec, gpu=gpu)
        
        log_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(log_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        
        logging.basicConfig(
            filename=join(self.cwd, 'gmx_log.txt'),
            filemode='w',
            format='%(asctime)s %(message)s',
            datefmt='%d/%m/%Y %I:%M:%S %p',
            level=log_level)
    
    @runmethod(name='Run Basic')
    def run_basic(self):
        #yield self.edit_mdp
        yield self.pdb2gmx
        yield self.edit_conf
        yield self.solvate
        yield self.neutralise
        yield self.minimise
        yield self.eq_nvt
        yield self.eq_npt
        yield self.run_md
        yield self.trjconv
        yield self.rmsd
    
    @runmethod(name='Testing')
    def testing(self):
        yield self.test

    @property
    def pdb(self):
        return self._pdb1

    def pre_process(
            self, chain='all', chain_exceptions=[], protein_only=True,
            water_exceptions=[], ions=['SO4', 'NA', 'CL', 'NI'], ion_exceptions=[]):
        """
        Quickly format .pdb file
        """
        if chain != 'all':
            self.isolate_chains(chain, exceptions=chain_exceptions)
        if protein_only:
            self.keep_protein_only()
        else:
            self.remove_waters(exceptions=water_exceptions)
            self.remove_ions(ions=ions, exceptions=ion_exceptions)

    def remove_waters(self, out_path='no_water.pdb', exceptions=[]):
        """
        Remove waters from active .pdb file
        
        Exceptions can be made by stating *molecule* number
        
        *6th column in .pdb file
        """
        with open(join(self.cwd, self.pdb)) as s:
            with open(join(self.cwd, out_path), 'w') as r:
                for line in s:
                    rec = line[:7]
                    if 'ATOM' in rec or 'HETATM' in rec or 'TER' in rec or 'END' in rec:
                        if 'HOH' not in line[15:22]:
                            r.write(line)
                        else:
                            for n in exceptions:
                                if n == int(line[22:26]):
                                    r.write(line)
                                    break

        self._pdb1 = out_path

    def remove_ions(self, out_path='no_ions.pdb', ions=[], exceptions=[]):
        """
        Remove ions from active .pdb file
        
        Exceptions can be made by stating *molecule* number
        
        *6th column in .pdb file
        """
        with open(join(self.cwd, self.pdb)) as s:
            with open(join(self.cwd, out_path), 'w') as r:
                for line in s:
                    rec = line[:7]
                    if 'ATOM' in rec or 'HETATM' in rec or 'TER' in rec or 'END' in rec:
                        for ion in ions:
                            if ion in line[15:22]:
                                break
                        else:
                            r.write(line)
                            continue

                        for n in exceptions:
                            if n == int(line[22:26]):
                                r.write(line)
                                break
        self._pdb = out_path

    def isolate_chains(self, out_path='single_chain.pdb', chain='A', exceptions=[]):
        """
        Keep only desired chain(s) in .pdb file
        """
        with open(join(self.cwd, self.pdb)) as s:
            with open(join(self.cwd, out_path), 'w') as r:
                for line in s:
                    rec = line[:7]
                    if 'ATOM' in rec or 'HETATM' in rec or 'TER' in rec or 'END' in rec:
                        if chain == line[21]:
                            r.write(line)
                        else:
                            for c in exceptions:
                                if c == line[21]:
                                    r.write(line)
                                    break
        self._pdb = out_path

    def keep_protein_only(self, out_path='protein_only.pdb'):
        """
        Removes all non-protein for .pdb file
        """
        with open(join(self.cwd, self.pdb)) as s:
            with open(join(self.cwd, out_path), 'w') as r:
                for line in s:
                    rec = line[:7]
                    if 'ATOM' in rec or 'TER' in rec or 'END' in rec:
                        r.write(line)
        self._pdb = out_path
    
    def convert_args(self, *args, **kw):
        """
        Converts method arguments to parse to gmx module
        
        e.g.: arg1 = value, arg2,... -> [-arg1, value, -arg2, ...]
        
        If arg contains a dash "-", replace it with a double
        underscore "__" as "-" cannot be in a key word arg in Python
        and some variables do use "_" within their name.
        """
        
        o = []
        for arg in args:
            arg = arg.replace('__', '-')
            o.append('-' + arg)

        for kwarg in kw.keys():
            k = kwarg.replace('__', '-')
            
            if isinstance(kw[kwarg], bool):
                if kw[kwarg]:
                    o.append('-' + k)
            else:
                o.append('-' + k)
                o.append(str(kw[kwarg]))
        return o
    
    
    def log(self, output, header=''):
        logging.info('========{}========\n'.format(header))
        logging.info('StdOut:\n')
        logging.info(str(output[0]))
        logging.info('StdError\n')
        logging.info(str(output[1]))
    
    def proc_cmd(self, cmd=[], inputs=[], cmd_name='', pc=[], nt=0, *a, **kw):
        """
        Process command
        """
        for i, j in enumerate(inputs):
            inputs[i] = str(j)
        
        remote = kw.pop('remote', False)
        no_prefix = kw.pop('no_prefix', False)
        cmd.extend(self.convert_args(*a, **kw))
        
        if not pc:
            cmds = []
        else:
            if not self.dry:
                # return if error in grompp
                return pc[0]
            else:
                cmds = pc[0]
        
        if not no_prefix:
            if not remote or not self.mpi:
                cmd = [GMX] + cmd
            else:
                cmd = [GMX_MPI] + cmd
        
        if self.dry or remote:
            if cmds:
                cmds = cmds + '\n'
            else:
                cmds = ''
            cmd = ' '.join(cmd)
            if nt:
                if not self.mpi or not remote:
                    cmd += ' -nt {}'.format(nt)
                else:
                    cmd = 'mpirun -np {0} -machinefile {1} '.format(NUM_PROCS, MACHINE_FILE) + cmd
            elif self.mpi and remote:
                cmd = 'mpirun -np 1 -machinefile {} '.format(MACHINE_FILE) + cmd
            if inputs:
                # use echo to pipe inputs to application
                ipts = 'echo {} | '.format(' '.join(str(i) for i in inputs))
                cmd = ipts + cmd
            return (cmds + cmd, remote)
        else:
            if nt:
                cmd += ['-nt', str(nt)]
        print(' '.join(cmd))
        proc = sp.Popen(
            cmd,
            stdin=sp.PIPE,
            stdout=sp.PIPE if self.quiet else sp.STDOUT,
            stderr=sp.PIPE if self.quiet else sp.STDOUT,
            cwd=self.cwd)
        if not inputs:
            output = proc.communicate()
        else:
            output = proc.communicate(input=b'%s' % '\n'.join(inputs))
        
        self.log(output, header=cmd_name)
            
        return proc.returncode
    
    def grompp(self, f='', c='', p='', o='', po='', t='', **kw):
        """
        GROMPP to create mdp run files
        """
        cmd = [
            GROMPP, '-f', f, '-c', c,
            '-p', p, '-o', o, '-po', po]
        if t:
            cmd.extend(['-t', t])
        remote = kw.pop('remote', False)
        return self.proc_cmd(cmd=cmd, cmd_name='GROMPP', remote=remote)
    
    def md_cmd(
            self, f, c, p, o, po, out,
            name='MDRUN', num_threads=6, t='', *a, **kw):
        """
        For generating MDRUN commands
        """
        rc = self.grompp(f, c, p, o, po, t=t, **kw)
        cmd = [
            MDRUN,
            '-s', o,
            '-deffnm', out]
        return self.proc_cmd(
            cmd=cmd, cmd_name=name, pc=rc, nt=num_threads, *a, **kw)
    
    def test(self, no_prefix=True, *a, **kw):
        return self.proc_cmd(
            cmd=['echo', 'hello'], cmd_name='Test', no_prefix=no_prefix, *a, **kw)
    
    def pdb2gmx(
            self, out='proc_pdb.gro', p='topol.top', i='posre.itp',
            water='spce', forcefield=15, auto_asign_his=False, missing=True,
            *a, **kw):
        self.topol = p
        add_args = []
        if auto_asign_his:
            his_states = []
            with open(join(self.cwd, self.pdb)) as pdb_file:
                pstate = '1'
                for line in pdb_file:
                    if 'HIS' == line[17:20]:
                        if 'ND1' == line[13:16]:
                            if 'N1+' == line[77:80]:
                                pstate = '2'
                            else:
                                pstate = '1'
                            his_states.append(pstate)

            his_cmd = '\n'.join(his_states)
            add_args.append('-his')
        
        cmd = [
            PDB2GMX,
            '-f', self.pdb,
            '-p', self.topol,
            '-i', i,
            '-o', out,
            '-water', water]
        if missing:
            cmd.append('-missing')
        
        if not auto_asign_his:
            inputs = [forcefield]
        else:
            inputs = [forcefield, his_cmd]
        
        return self.proc_cmd(cmd=cmd, inputs=inputs, cmd_name='PDB TO GMX', *a, **kw)

    
    def edit_conf(
            self, gro='proc_pdb.gro', out='new_box.gro', bt='cubic',
            d=1.0, *a, **kw):
        cmd = [
            EDITCONF,
            '-f', gro,
            '-o', out,
            '-c',
            '-d', str(d),
            '-bt', bt]
        return self.proc_cmd(cmd=cmd, cmd_name='EDIT CONF', *a, **kw)

    
    def solvate(
            self, cp='new_box.gro', out='new_solv.gro', p='topol.top',
            cs='spc216.gro', *args, **kw):
        """
        solvate box
        """
        cmd = [
            SOLVATE,
            '-cp', cp,
            '-cs', cs,
            '-o', out,
            '-p', p]
        return self.proc_cmd(cmd=cmd, cmd_name='SOLVATE', *args, **kw)
        
    
    def neutralise(
            self, f='ions.mdp', c='new_solv.gro', p='topol.top',
            o='ions.tpr', po='mdout.mdp', out='solv_ions.gro',
            pname='NA', nname='CL', conc=0.1, com=13, neutral=True,
            *a, **kw):
        """
        neutralise box by adding ions
        """
        rc = self.grompp(f, c, p, o, po, **kw)
        cmd = [
            GENION,
            '-s', o,
            '-o', out,
            '-p', p,
            '-pname', pname,
            '-nname', nname,
            '-conc', str(conc)]
        if neutral:
            cmd += ['-neutral']
        inputs = [com]
        return self.proc_cmd(
            cmd=cmd, inputs=inputs, cmd_name='NEUTRALISE', pc=rc, *a, **kw)

   
    def minimise(self, *args, **kw):
        defaults = dict(
            f='minim.mdp',
            c='solv_ions.gro',
            p=self.topol,
            o='em.tpr',
            po='mdout.mdp',
            out='em',
            num_threads=self.num_threads)
        defaults.update(kw)
        return self.md_cmd(name='MINIMISE', *args, **defaults)
    
    def eq_nvt(self, *args, **kw):
        defaults = dict(
            f='nvt.mdp',
            c='em.gro',
            p=self.topol,
            o='nvt.tpr',
            po='mdout.mdp',
            out='nvt',
            num_threads=self.num_threads)
        defaults.update(kw)
        return self.md_cmd(name='EQUILIBRIATE NVT', *args, **defaults)
    
    def eq_npt(self, *args, **kw):
        defaults = dict(
            f='npt.mdp',
            c='nvt.gro',
            p=self.topol,
            o='npt.tpr',
            po='mdout.mdp',
            t='nvt.cpt',
            out='npt',
            num_threads=self.num_threads)
        defaults.update(kw)
        return self.md_cmd(name='EQUILIBRIATE NPT', *args, **defaults)
    
    def run_md(self, *args, **kw):
        """
        run MD simulation
        """
        defaults = dict(
            f='md.mdp',
            c='npt.gro',
            p=self.topol,
            o='md_0_1.tpr', 
            po='mdout.mdp',
            t='npt.cpt',
            out='md_0_1',
            num_threads=self.num_threads)
        defaults.update(kw)
        return self.md_cmd(name='MD RUN', *args, **defaults)
    
    
    def restart(
            self, s='md_0_1.tpr', cpi='md_0_1.cpt', out='md_0_1', append=True,
            num_threads=6, *a, **kw):
        """
        Restart failed/incomplete runs
        """
        cmd = [
            MDRUN,
            '-s', s,
            '-cpi', cpi,
            '-deffnm', out]
        if append:
            cmd += ['-append']
        else:
            cmd += ['-noappend']
        
        return self.proc_cmd(
            cmd=cmd, cmd_name='MD RUN RESTART', nt=num_threads, *a, **kw)
        
    
    def tpbconv(
            self, s='md_0_1.tpr', extend='', until='', nsteps='', time='',
            o='md_0_1.tpr', *a, **kw):
        """
        Convert tpr file to extend/adjust run paremeters
        """
        cmd = [
            TPBCONV,
            '-s', s,
            '-o', o]
        
        if extend:
            cmd += ['-extend', extend]
        if until:
            cmd += ['-until', until]
        if nsteps:
            cmd += ['-nsteps', nsteps]
        if time:
            cmd += ['time', time]
        
        return self.proc_cmd(cmd=cmd, cmd_name='TPB CONVERT', *a, **kw)
    
    
    def trjconv(
            self, s='md_0_1.tpr', f='md_0_1.xtc', o='centered.gro',
            pbc='mol', ur='compact', center=True, center_group=1,
            out_group=0, *a, **kw):
        """
        Convert trajectory to remove pbc
        """
        inputs = []
        cmd = [
            TRJCONV,
            '-s', s,
            '-f', f,
            '-o', o,
            '-pbc', pbc,
            '-ur', ur] 
        if center:
            cmd.append('-center')
            inputs.append(center_group)
        inputs.append(out_group)        
        return self.proc_cmd(cmd=cmd, inputs=inputs, cmd_name='REMOVE PBC', *a, **kw)
    
    
    def rmsd(
            self, s='md_0_1.tpr', f='centered.gro', o='RMSDgraph.xvg',
            tu='ns', fit_group1=4, fit_group2=4, *a, **kw):
        """
        Calculate RMSDs for completed run
        
        If using centered.gro with limmited output groups, e.g. protein
        only (group=1), the fit groups 1 and 2 must also be present in
        the input groups.
        
        Typical group list: 
        
        Group     0 (         System)
        Group     1 (        Protein)
        Group     2 (      Protein-H)
        Group     3 (        C-alpha)
        Group     4 (       Backbone)
        Group     5 (      MainChain)
        Group     6 (   MainChain+Cb)
        Group     7 (    MainChain+H)
        Group     8 (      SideChain)
        Group     9 (    SideChain-H)
        Group    10 (    Prot-Masses)
        Group    11 (    non-Protein)
        Group    12 (          Water)
        Group    13 (            SOL)
        Group    14 (      non-Water)
        Group    15 (            Ion)
        Group    16 (             NA)
        Group    17 (             CL)
        Group    18 ( Water_and_ions)
        """
        inputs = [fit_group1, fit_group2]
        cmd = [
            RMSD,
            '-s', s,
            '-f', f,
            '-o', o,
            '-tu', tu]
        
        return self.proc_cmd(cmd=cmd, inputs=inputs, cmd_name='RMSD', *a, **kw)

    
    def clean_dir(self):
        """
        Remove any old bakup files - any files starting and ending
        with a "#"
        """
        o = ['', '']
        for f in glob.glob(join(self.cwd, '*.*')):
            if basename(f).startswith('#') and f.endswith('#'):
                if not self.quiet:
                    print('Removing: ' + f)
                o[0] += '\nRemoving: ' + f
                os.remove(f)
        self.log(o, header='Cleaning up...')
        
    def save_scripts(self):
        if self.local_script:
            with open(join(self.cwd, 'local'), 'wb') as s:
                s.write(bytes(self.local_script))
        if self.qsub:
            with open(join(self.cwd, 'gmxqs'), 'wb') as s:
                s.write(bytes(self.qsub))

           
def edit_top(topo, old_include='___', new_include=''):
    out = ''
    with open(topo) as s:
        for line in s:
            if '#include' == line[:8]:
                if old_include in line:
                    line = '#include "{0}"\n'.format(new_include)
            out += line

    with open(topo, 'w') as s:
        s.write(out)

def edit_mdp(file_path, **kw):
    """
    Change one or more arguments contained within an .mdp file
    
    If arg contains a dash "-", replace it with a double
    underscore "__"
    
    Add " ; ..." to the end of a value to add an in-line comment.
    """

    lines = []
    with open(file_path) as r:
        for line in r:
            lines.append(line)
            if ';' in line:
                line_args = line[:line.find(';')]
            else:
                line_args = line
            for kwarg in kw.keys():
                if kwarg.replace('__', '-') in line_args:
                    if str(kw[kwarg]) == 'DEL':
                        lines[-1] = '\n' # delete line
                    else:
                        lines[-1] = '{0} = {1}\n'.format(
                            kwarg.replace('__', '-'),
                            str(kw[kwarg]))
                    del kw[kwarg]
    
    # if arg cannot be found in file, append to end
    for kwarg in kw.keys():
        lines.append('{0} = {1}\n'.format(
            kwarg.replace('__', '-'),
            str(kw[kwarg])))

    with open(file_path, 'w') as w:
        for line in lines:
            w.write(line)

    return 0


def make_mdp(file_path, **kw):
    """
    Creates a new .mdp file
    
    If arg contains a dash "-", replace it with a double
    underscore "__"
    
    All available options and example values:
    
    VARIOUS PREPROCESSING OPTIONS:
    
    title                    = deprecated - will be ignored
    cpp                      = /lib/cpp
    include                  = -I../top
    define                   = 
    
    RUN CONTROL PARAMETERS:
    
    integrator               = md
    start time and timestep in ps:
    tinit                    = 0
    dt                       = 0.002
    nsteps                   = 500000
    number of steps for center of mass motion removal:
    nstcomm                  = 1
    comm-grps                = 
    
    LANGEVIN DYNAMICS OPTIONS:
    
    Temperature, friction coefficient (amu/ps) and random seed:
    bd-temp                  = 300
    bd-fric                  = 0
    ld-seed                  = 1993
    
    ENERGY MINIMIZATION OPTIONS:
    
    Force tolerance and initial step-size:
    emtol                    = 100
    emstep                   = 0.01
    Max number of iterations in relax-shells:
    niter                    = 20
    Frequency of steepest descents steps when doing CG:
    nstcgsteep               = 1000
    
    OUTPUT CONTROL OPTIONS:
    
    Output frequency for coords (x), velocities (v) and forces (f):
    nstxout                  = 5000
    nstvout                  = 5000
    nstfout                  = 0
    Output frequency for energies to log file and energy file:
    nstlog                   = 5000
    nstenergy                = 250
    Output frequency and precision for xtc file:
    nstxout-compressed       = 250
    compressed-x-precision   = 1000
    This selects the subset of atoms for the xtc file. You can
    select multiple groups. By default all atoms will be written:
    compressed-x-grps        = Protein
    Selection of energy groups:
    energygrps               = Protein  SOL
    
    NEIGHBORSEARCHING PARAMETERS:
    
    nblist update frequency:
    nstlist                  = 10
    ns algorithm (simple or grid):
    ns-type                  = grid
    Periodic boundary conditions: xyz or none:
    pbc                      = xyz
    nblist cut-off:
    rlist                    = 0.8
    
    OPTIONS FOR ELECTROSTATICS AND VDW:
    
    Method for doing electrostatics:
    coulombtype              = cut-off
    rcoulomb-switch          = 0
    rcoulomb                 = 1.4
    Dielectric constant (DC) for cut-off or DC of reaction field:
    epsilon-r                = 1
    Method for doing Van der Waals:
    vdw-type                 = Cut-off
    cut-off lengths:
    rvdw-switch              = 0
    rvdw                     = 0.8
    Apply long range dispersion corrections for Energy and Pressure:
    DispCorr                 = No
    Spacing for the PME/PPPM FFT grid:
    fourierspacing           = 0.12
    FFT grid size, when a value is 0 fourierspacing will be used:
    fourier-nx               = 0
    fourier-ny               = 0
    fourier-nz               = 0
    EWALD/PME/PPPM parameters:
    pme-order                = 4
    ewald-rtol               = 1e-05
    epsilon-surface          = 0
    
    OPTIONS FOR WEAK COUPLING ALGORITHMS:
    
    Temperature coupling:
    tcoupl                   = Berendsen
    Groups to couple separately:
    tc-grps                  = Protein      SOL
    Time constant (ps) and reference temperature (K):
    tau-t                    = 0.1  0.1
    ref-t                    = 300  300
    Pressure coupling:
    Pcoupl                   = Berendsen
    Pcoupltype               = Isotropic
    Time constant (ps), compressibility (1/bar) and reference P (bar):
    tau-p                    = 1.0
    compressibility          = 4.5e-5
    ref-p                    = 1.0
    
    SIMULATED ANNEALING CONTROL:
    
    annealing                = no
    Time at which temperature should be zero (ps):
    zero-temp-time           = 0
    
    GENERATE VELOCITIES FOR STARTUP RUN: 
    
    gen-vel                  = yes
    gen-temp                 = 300
    gen-seed                 = 173529
    
    OPTIONS FOR BONDS:
    
    constraints              = all-bonds
    Type of constraint algorithm:
    constraint-algorithm     = Lincs
    Do not constrain the start configuration:
    unconstrained-start      = no
    Relative tolerance of shake:
    shake-tol                = 0.0001
    Highest order in the expansion of the constraint coupling matrix:
    lincs-order              = 4
    Lincs will write a warning to the stderr if in one step a bond
    rotates over more degrees than:
    lincs-warnangle          = 30
    Convert harmonic bonds to morse potentials:
    morse                    = no
    
    NMR refinement stuff
    Distance restraints type: No, Simple or Ensemble:
    disre                    = No
    Force weighting of pairs in one distance restraint: Equal or 
    Conservative:
    disre-weighting          = Equal
    Use sqrt of the time averaged times the instantaneous violation:
    disre-mixed              = no
    disre-fc                 = 1000
    disre-tau                = 0
    Output frequency for pair distances to energy file:
    nstdisreout              = 100
    
    Free energy control stuff:
    free-energy              = no
    init-lambda              = 0
    delta-lambda             = 0
    sc-alpha                 = 0
    sc-sigma                 = 0.3
    
    Non-equilibrium MD stuff:
    acc-grps                 = 
    accelerate               = 
    freezegrps               = 
    freezedim                = 
    cos-acceleration         = 0
    energygrp-excl           =
    
    Electric fields
    Format is number of terms (int) and for all terms an amplitude (real)
    and a phase angle (real):
    E-x                      = 
    E-xt                     = 
    E-y                      = 
    E-yt                     = 
    E-z                      = 
    E-zt                     = 
    
    User defined thingies:
    user1-grps               = 
    user2-grps               = 
    userint1                 = 0
    userint2                 = 0
    userint3                 = 0
    userint4                 = 0
    userreal1                = 0
    userreal2                = 0
    userreal3                = 0
    userreal4                = 0
    """
    with open(file_path, 'w') as s:
        for kwarg in kw.keys():
            s.write(kwarg.replace('__', '-') + ' = ' + str(kw[kwarg]) + '\n')
    return 0


if __name__ == '__main__':
    USER = 'user'
    test = Gmx('mol.pdb', cwd='../../md_params', remote=True, mpi=True, ssh_key='', hrs=24, num_nodes=1, gpu=1)
    run = test.run_basic(
        all_remote=True,
        trjconv=dict(center_group=1, out_group=1),
        rmsd=dict(fit_group1=4, fit_group2=4))

