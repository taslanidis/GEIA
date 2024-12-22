# GEIA (Gradient Embedding Inversion Attack)
A reproducibility study and extension of GEIA - Information Retrieval 2 Course @ UvA

## Getting Started

### 1. Baseline Models
1. Install environment:
```bash
sbatch scripts/jobs/install_env_locally.job
```

2. Run baseline evaluation:
```bash
bash scripts/bashscripts/launch_baseline_eval.sh
```

### 2. GEIA Attacker Implementation
1. Install GEIA environment:
```bash
sbatch scripts/jobs/install_env_extension.job
```

2. Train and evaluate the attacker:
```bash
# Train the GEIA attacker
bash scripts/bashscripts/launch_geia_qnli_train_random_gpt_medium.sh

# Evaluate the attacker
bash scripts/bashscripts/launch_geia_qnli_eval_random_gpt_medium.sh
```

### 3. Training Data Leakage Extension
This extension investigates leakage from training data using LLM reasoners.

1. Set up LLM environment:
```bash
cd LLM_instruct_masking/
sbatch install_env_extension.job
```

2. Download LLM weights (e.g., GLM-4):
```bash
cd LLM_instruct_masking/
sbatch download_glm-4.job
```

3. Run the analysis pipeline:
```bash
# Step 1: Generate masks and alternative sentences (10% of dataset)
cd LLM_instruct_masking/
sbatch run_masking.job

# Step 2: Calculate log-probabilities
# Note: Requires GEIA GPT-2 attacker trained on Personachat dataset
sbatch scripts/jobs/detect_train_leakage.job

# Step 3: Perform statistical analysis
sbatch scripts/jobs/detect_dist_difference_leakage.job
```

## Project Structure
- `scripts/jobs/`: Contains SLURM job scripts
- `scripts/bashscripts/`: Contains bash execution scripts
- `LLM_instruct_masking/`: Folder for masking and alternative sentences generation  with LLM reasoners

## Notes
- All SLURM jobs are designed to run on a cluster environment
- The leakage analysis requires a pre-trained GEIA GPT-2 attacker model
- Results of the leakage analysis are stored in the `logs/` directory

