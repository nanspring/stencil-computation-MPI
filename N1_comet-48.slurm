#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=apf-strong-scale-48
#SBATCH --output="apf-strong-scale-48.out"
#SBATCH --partition="compute"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00
# Commands go here

#ibrun -np 480 ./apf -n 1800 -i 2000 -x 20 -y 24

# N1
ibrun -np 24 ./apf -n 1800 -i 2000 -x 2 -y 12 
ibrun -np 48 ./apf -n 1800 -i 2000 -x 2 -y 24 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 2 -y 48 

# N2
# ibrun -np 96 ./apf -n 2000 -i 8000 -x 4 -y 24 -k
# ibrun -np 128 ./apf -n 2000 -i 8000 -x 4 -y 32 -k
# ibrun -np 240 ./apf -n 2000 -i 8000 -x 4 -y 60 -k
# ibrun -np 384 ./apf -n 2000 -i 8000 -x 4 -y 96
# ibrun -np 480 ./apf -n 2000 -i 8000 -x 4 -y 120

#ibrun -np 48 ./apf -n 1800 -i 2000 -x 4 -y 12
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 6 -y 8
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 8 -y 6
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 12 -y 4
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 24 -y 2