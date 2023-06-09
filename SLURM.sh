#!/bin/bash
#SBATCH --job-name=Unet          
#SBATCH --nodes=1        
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=4 
#SBATCH --partition=long
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --time=7:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sherz22@ku.edu.tr    
module load python/3.9.5
module load cuda/11.4
python main.py
