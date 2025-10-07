#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-609
#SBATCH -o /mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/training/slurm_outs/%x_%j.out
#SBATCH -t 0-00:10:00
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --nodes 1

export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 hostname 2>&1 | head -n1)
export MASTER_PORT=29501

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=1 \
    --node_rank=$SLURM_PROCID \
    launch.py --config dinov3


# torchrun --nproc_per_node=4 --master-port=29501 launch.py --config dinov3
