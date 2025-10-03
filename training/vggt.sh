#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-609
#SBATCH -o /mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/training/slurm_outs/%x_%j.out
#SBATCH -t 0-10:00:00
#SBATCH --gpus-per-node=A100:4
#SBATCH --nodes 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davnords@chalmers.se


torchrun --nproc_per_node=4 launch.py --config dinov3
