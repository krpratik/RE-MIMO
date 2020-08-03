#!/bin/bash
#SBATCH --job-name=validate_blast
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=60000M
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kr.pratik73@gmail.com

#module purge

module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3


srun python3 test_blast.py
