#!/bin/bash
#SBATCH --output=output_%A_%a.log        # Output file for logs (%j will append the job ID)
#SBATCH --error=error_%A_%a.log          # Error file for logs (%j will append the job ID)
#SBATCH --ntasks=1                    # Run on a single CPU core
#SBATCH --cpus-per-task=1             # Number of CPU cores per task (modify based on your requirement)
#SBATCH --mem=15G                     # Memory allocation (modify based on your model's needs)
#SBATCH --mail-type=END,FAIL          # Notify user by email when the job ends or fails

# Run the Python script
python3 run_horseshoe_snp.py $sampler $SLURM_ARRAY_TASK_ID $proposal_sd
