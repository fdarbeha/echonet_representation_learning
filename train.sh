#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --account=def-wanglab
#SBATCH --job-name=pastssl-ordered-pairs
#SBATCH --output=%x-%j.out

#SBATCH --gres=gpu:4	# Number of GPU(s) per node
#SBATCH --mem=15G		# memory per node
#SBATCH --cpus-per-task=8	# CPU cores/threads

python main.py --mode pastssl --type PaSTssl --similarity bilinear --batch_size 70 --run 2
#python /home/fdarbeha/scratch/echonet_representation_learning/main.py --mode cpc --type 3d_cpc --batch_size 10 --similarity cosine --run 1
