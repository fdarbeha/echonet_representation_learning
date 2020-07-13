#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --account=def-wanglab
#SBATCH --job-name=echo-net-cosine
#SBATCH --output=%x-%j.out

#SBATCH --gres=gpu:4	# Number of GPU(s) per node
#SBATCH --mem=10G		# memory per node
#SBATCH --cpus-per-task=8	# CPU cores/threads

python /home/fdarbeha/scratch/echonet_representation_learning/main.py --mode ssl --batch_size 60 --similarity cosine