#!/bin/bash
#SBATCH --job-name=PRED_ATTACK
#SBATCH --error=PRED_ATTACK-%A-%a.err
#SBATCH --output=PRED_ATTACK-%A-%a.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-1

module purge
module load Python

source deactivate
source activate env_humor

array=(one_liners-693 pun_of_the_day-23 semeval_2017_task_7-693 short_jokes-453 reddit_jokes_last_laught-693 semeval_2021_task_7-47 funlines_and_human_microedit_paper_2023-23 unfun_me-23 the_naughtyformer-693 meta_dataset-977)
hd_idx=$SLURM_ARRAY_TASK_ID

srun python3 -u run_attack.py ${array[$hd_idx]}
