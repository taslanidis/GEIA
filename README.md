<h1 align="center">Information Leakage of Sentence Embeddings via GEIA (Gradient Embedding Inversion Attack) 
  <strong><a href="https://arxiv.org/abs/2504.16609" target="_blank">🗒️Paper</a></strong> </h1> 

<p align="center">
  <strong> Antonios Tragoudaras</strong><sup>*</sup> &nbsp; | &nbsp;
  <strong> Theofanis Aslanidis</strong><sup>*</sup> &nbsp; | &nbsp;
  <strong> Emmanouil Georgios Lionis</strong><sup>*</sup> &nbsp; | &nbsp;
  <strong> Marina Orozco González</strong><sup>*</sup> &nbsp; | &nbsp;
  <strong> Panagiotis Eustratiadis </strong>
</p>

<p align="center"><sup>*</sup>These authors contributed equally</p>

### <p align="center"><strong>SIGIR2025</strong></p>





## Getting Started

### 1. Baseline Models
1. Install environment:
```bash
sbatch scripts/jobs/install_env_locally.job
```

2. Reproduce the baseline models (MLC & MSP):
```bash
bash scripts/bashscripts/launch_baseline_eval.sh
```

### 2. GEIA Attacker Implementation
1. Use the same environment as the baseline models:

2. Train and evaluate the attacker:
```bash
# Train the GEIA attacker
bash scripts/bashscripts/launch_geia_qnli_train_random_gpt_medium.sh

# Evaluate the attacker
bash scripts/bashscripts/launch_geia_qnli_eval_random_gpt_medium.sh
```

### 3. Training Data Leakage Extension - RQ: Do sentence embeddings leak sensitive information
from the training data?

1. Set up LLM reasoner environment:
```bash
cd LLM_instruct_masking/
sbatch install_local_LLM_env.job
```

2. Download LLM reasoner weights (e.g., GLM-4):
```bash
cd LLM_instruct_masking/
sbatch download_glm-4.job
```

3. Produce the masks and alternative sentences with the LLM reasoner:
```bash

cd LLM_instruct_masking/
sbatch run_masking.job
```

4. Calculate the log-probabilities of the masks and alternative sentences with the GEIA attacker:
```bash
# Step 1: Install extension environemt
sbatch scripts/jobs/install_env_extension.job
# Step 2: Calculate log-probabilities
# Calculating & stores the log-probs of the mask and alternative sentences with and without the sentence embeddinghs with differen vicitim models. This requires the GEIA gpt-2 attcker model to be trained on the Personachat dataset.
# Note: Requires GEIA GPT-2 attacker trained on Personachat dataset
sbatch scripts/jobs/detect_train_leakage.job

# Step 3: Perform statistical analysis
# Identifies the mean of the populatiuon and perfromas signifcance tests, based on the leakage log-probs stored in the `logs/` folder.
sbatch scripts/jobs/detect_dist_difference_leakage.job
```

### 4. Conversational Attack - RQ: In a conversational setting, can GEIA reconstruct
the input text that prompted an LLM, based on the LLM’s
responses?

1. Use the same environment as the baseline models:

2. Train the sentence-encoder model:
```bash
# Step 1: Checkout to the settence_encoder folder
git checkout sentence_encoder
# Step 2: Train the sentence-encoder model
sbatch LLM_test_with_trained_sentence_embeddings.job
# Step 3: Evaluate the sentence-encoder model
sbatch LLM_test_with_trained_sentence_embeddings_eval.job
```
3.  Witout training:
```bash
# Step 1: Checkout to the settence_encoder folder
git checkout LLM-addition
# Step 2: Evaluate the sentence-encoder model
sbatch LLM_train.job
# Step 3: Evaluate the sentence-encoder model
sbatch LLM_eval.job
```

## Project Structure
- `scripts/jobs/`: Contains SLURM job scripts
- `scripts/bashscripts/`: Contains bash execution scripts
- `LLM_instruct_masking/`: Folder for masking and alternative sentences generation  with LLM reasoners


