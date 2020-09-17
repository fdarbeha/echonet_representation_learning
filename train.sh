#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --account=def-wanglab
#SBATCH --job-name=echo-cpc-tau1-3pos
#SBATCH --output=%x-%j.out

#SBATCH --gres=gpu:1	# Number of GPU(s) per node
#SBATCH --mem=15G		# memory per node
#SBATCH --cpus-per-task=8	# CPU cores/threads

python main.py --mode ssl --type PaSTSSL --batch_size 20 --run 1
#python /home/fdarbeha/scratch/echonet_representation_learning/main.py --mode cpc --type 3d_cpc --batch_size 10 --similarity cosine --run 1
