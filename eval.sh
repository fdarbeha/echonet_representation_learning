#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-wanglab
#SBATCH --job-name=eval-ours-tsne
#SBATCH --output=%x-%j.out

#SBATCH --gres=gpu:1	# Number of GPU(s) per node
#SBATCH --mem=10G		# memory per node
#SBATCH --cpus-per-task=8	# CPU cores/threads

python /home/fdarbeha/scratch/echonet_representation_learning/main.py --eval True --mode tsne --type 3d --run 3 --checkpoint 200 --batch_size 20
