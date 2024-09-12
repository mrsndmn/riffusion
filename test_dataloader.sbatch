#!/bin/env bash
#SBATCH --job-name="train_clap_prior_config"
#SBATCH --ntasks=1              # Количество MPI процессов
#SBATCH --ntasks-per-node=1     # Количество процессов на один узел
#SBATCH --cpus-per-task=8       # Количество ядер CPU, выделенных для одного процесса
#SBATCH -C type_a|type_b|type_c
#SBATCH --time-min=30

module purge
module load Python

source deactivate
source activate audio_dalle2
nvidia-smi

set -eo pipefail
set -o xtrace

#Executable


export PYTHONPATH=/home/dtarasov/workspace/hse-audio-dalle2/riffusion:/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src

srun python3 ./test_dataloader.py \
    --pretrained_model_name_or_path='riffusion/riffusion-model-v1' \
    --train_data_dir='/home/dtarasov/workspace/hse-audio-dalle2/data/audiocaps_train_spectrograms_1k/' \
    --train_batch_size 1