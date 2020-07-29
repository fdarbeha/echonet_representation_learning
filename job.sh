#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --account=def-wanglab
#SBATCH --job-name=echo-bilinear-tau1-1pos
#SBATCH --output=%x-%j.out

#SBATCH --gres=gpu:4	# Number of GPU(s) per node
#SBATCH --mem=15G		# memory per node
#SBATCH --cpus-per-task=8	# CPU cores/threads

python /home/fdarbeha/scratch/echonet_representation_learning/main.py --mode ssl --type 3d --batch_size 40 --similarity bilinear --run 1