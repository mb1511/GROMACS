#!/bin/bash
# request resources
#PBS -l nodes=1:ppn=16:gpus=1
#PBS -l walltime=24:00:00

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
cat $PBS_NODEFILE > machine.file.$PBS_JOBID
numnodes=`wc $PBS_NODEFILE | awk '{print $1}'`

echo 15 | mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi pdb2gmx -f protein.pdb -p topol.top -i posre.itp -o proc_pdb.gro -water spce -missing
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi editconf -f proc_pdb.gro -o new_box.gro -c -d 1.0 -bt cubic
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi solvate -cp new_box.gro -cs spc216.gro -o new_solv.gro -p topol.top
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi grompp -f ions.mdp -c new_solv.gro -p topol.top -o ions.tpr -po mdout.mdp
echo 13 | mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -conc 0.1 -neutral
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -po mdout.mdp
mpirun -np numnodes -machinefile machine.file.$PBS_JOBID gmx_mpi mdrun -s em.tpr -deffnm em
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -po mdout.mdp
mpirun -np numnodes -machinefile machine.file.$PBS_JOBID gmx_mpi mdrun -s nvt.tpr -deffnm nvt
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi grompp -f npt.mdp -c nvt.gro -p topol.top -o npt.tpr -po mdout.mdp -t nvt.cpt
mpirun -np numnodes -machinefile machine.file.$PBS_JOBID gmx_mpi mdrun -s npt.tpr -deffnm npt
mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi grompp -f md.mdp -c npt.gro -p topol.top -o md_0_1.tpr -po mdout.mdp -t npt.cpt
mpirun -np numnodes -machinefile machine.file.$PBS_JOBID gmx_mpi mdrun -s md_0_1.tpr -deffnm md_0_1
echo 1 1 | mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi trjconv -s md_0_1.tpr -f md_0_1.xtc -o centered.gro -pbc mol -ur compact -center
echo 4 4 | mpirun -np 1 -machinefile machine.file.$PBS_JOBID gmx_mpi rms -s md_0_1.tpr -f centered.gro -o RMSDgraph.xvg -tu ns
