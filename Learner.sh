#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M               
#SBATCH --time=00:30:00

module load python/3.9

cd $SCRATCH
cp -R $SCRATCH/Transitions $SLURM_TMPDIR
cp -R $SCRATCH/Models $SLURM_TMPDIR
cd $HOME

source OPMGen/bin/activate

wandb login a7f822756af06f4332c2c78d8399c6a2b7f856cc

python OPMLearner/Train_main.py

cd $SLURM_TMPDIR
mv $SLURM_TMPDIR/Model1 $SCRATCH/Models