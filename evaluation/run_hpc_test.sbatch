#!/bin/bash

#SBATCH --job-name=llm-judge
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#SBATCH --partition=kisski
#SBATCH --nodes=1               
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A100:1
#SBATCH --time=48:00:00
#SBATCH --mem=80G     
#SBATCH -C inet    
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=mansour.abeer@mh-hannover.de

# node the job ran on + empty line
echo "Job ran on:" $(hostname)
echo "" 
module load gcc/13.2.0
module load cuda/12.6.2

export http_proxy=http://www-cache.gwdg.de:3128
export https_proxy=http://www-cache.gwdg.de:3128
export ftp_proxy=http://www-cache.gwdg.de:3128

source /projects/extern/kisski/l3s_research_llm_training/dir.project/abeer/miniconda3/etc/profile.d/conda.sh
conda activate arena2



python llm_as_judge.py 

wait

echo ""
echo "Job finished"  

# job will finish automatically as soon as the script has finished running, regardless of the specified job time.
