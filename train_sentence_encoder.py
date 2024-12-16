from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.training_args import BatchSamplers

from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, Dataset as Dataset_datasets 
import torch
import numpy as np
import argparse
import wandb
from data_process import get_sent_list
from dataset import original_dataset, LLM_dataset

model_cards = {}
model_cards['sent_t5_base'] = 'sentence-t5-base'
model_cards['sent_roberta'] = 'all-roberta-large-v1'

model_cards["meta-llama"] = "meta-llama/Meta-Llama-3-8B"
model_cards["meta-llama2-7b"] = "meta-llama/Llama-2-7b-chat-hf"

parser = argparse.ArgumentParser(
    description="Training external decoder to attack an LLM"
)
parser.add_argument(
    "--model_dir",
    type=str,
    default="microsoft/DialoGPT-large",
    help="Dir of your model",
)
parser.add_argument("--num_epochs", type=int, default=10, help="Training epoches.")
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=40,
    help="How many tokens should the LLM use to answer the prompt?",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch_size #.")
parser.add_argument(
    "--dataset",
    type=str,
    default="fingpt-sentiment",
    help="Name of dataset: personachat or qnli or fingpt-sentiment",
)
parser.add_argument(
    "--embed_model",
    type=str,
    default="meta-llama2-7b",
    help="Name of embedding model: meta-llama2-7b/meta-llama",
)
parser.add_argument(
    "--decode",
    type=str,
    default="beam",
    help="Name of decoding methods: beam/sampling",
)
parser.add_argument(
    "--sentence_aggregation",
    type=str,
    default="sentence-t5-base",
    help="Name of sentence embending creation methods: sent_roberta/sent_t5_base/trained_sent_roberta",
)
parser.add_argument(
    "--save_embedding",
    type=str,
    default="./LLM_hidden_states/",
    help="The path to save the LLM last hidden states",
)
parser.add_argument(
    "--percentage_of_dataset",
    type=int,
    default=1,
    help="how much of the dataset do you want to upload",
)
parser.add_argument("--student_model", type=str, default="all-roberta-large-v1")  
parser.add_argument("--teacher_model", type=str, default="all-roberta-large-v1")
parser.add_argument(
    "--seed",
    type=int,
    default=14,
    help="What should be the seed",
)
args = parser.parse_args()

config = {}
config["model_dir"] = args.model_dir
config["num_epochs"] = args.num_epochs
config["batch_size"] = args.batch_size
config["dataset"] = args.dataset
config["embed_model"] = args.embed_model
config["decode"] = args.decode
config["embed_model_path"] = model_cards[config["embed_model"]] if config["embed_model"] in model_cards else config["embed_model"]
config["sentence_aggregation"] = model_cards[args.sentence_aggregation] if args.sentence_aggregation in model_cards else args.sentence_aggregation
config["max_new_tokens"] = args.max_new_tokens
config["seed"] = args.seed
config['use_opt'] = False
config["last_hidden_states_flag"] = False
config["data_type"] = "train"

# custom save paths
attacker_model: str = (
    "rand_gpt2_m" if args.model_dir == "random_gpt2_medium" else "dialogpt2"
)

config["save_embedding"]: str = f"{args.save_embedding}/saved_embeddings_{config['dataset']}_{config['embed_model']}_{config['max_new_tokens']}_{config['data_type']}_{args.percentage_of_dataset}"

attacker_save_name: str = (
    f"attacker_{attacker_model}_{config['dataset']}_{config['embed_model']}_{config['sentence_aggregation']}_{config['max_new_tokens']}"
)
projection_save_name: str = (
    f"projection_{attacker_model}_{config['dataset']}_{config['embed_model']}_{config['sentence_aggregation']}_{config['max_new_tokens']}"
)
sentence_embedding_model_save_name: str = (
    f"sentence_embedding_model_{attacker_model}_{config['dataset']}_{config['embed_model']}_{config['sentence_aggregation']}_{config['max_new_tokens']}"
)
config["attacker_save_path"] = f"models/{attacker_save_name}"
config["projection_save_path"] = f"models/{projection_save_name}"
config["sentence_embedding_model_save_path"] = (
    f"models/{sentence_embedding_model_save_name}"
)
config["attacker_save_name"] = attacker_save_name
config["projection_save_name"] = projection_save_name
config["student_model"] = args.student_model
config["teacher_model"] = args.teacher_model

if torch.cuda.is_available():
    config["device"] = torch.device("cuda:0")
else:
    raise ImportError(
        "Torch couldn't find CUDA devices. Can't run the attacker on CPU."
    )

print("args:", args)
print("config:",config)
wandb.init(project="GEIA", name = "sentence_encoder", config=config)

# Load the dataset
sent_list = get_sent_list(config)
the_original_dataset: Dataset = original_dataset(sent_list)
# reduce the original_dataset to just args.percentage_of_dataset samples to check the code
the_original_dataset = torch.utils.data.Subset(the_original_dataset, range(np.floor(len(the_original_dataset) * args.percentage_of_dataset/100).astype(int)))
the_LLM_dataset: Dataset = LLM_dataset(the_original_dataset, config)

train_dataset: DatasetDict = the_LLM_dataset.convert_to_dataset_dict()

teacher_model = SentenceTransformer(args.teacher_model)
def get_teacher_output(batch):
    with torch.no_grad():
        return {"label":teacher_model.encode(batch["input_LLM"])}

train_dataset = train_dataset.map(get_teacher_output, batched=True, batch_size=args.batch_size*4)
del teacher_model 
torch.cuda.empty_cache()

# Load the model
student_model = SentenceTransformer(args.student_model)

# Define the train loss
train_loss = losses.MSELoss(model=student_model)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 5. Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=config["sentence_embedding_model_save_path"],
    # Optional training parameters:
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_bf16_supported() == False,  # Enable fp16 if bf16 is not supported
    bf16=torch.cuda.is_bf16_supported(),  # Enable bf16 only if supported
    dataloader_num_workers=4,  # Increase DataLoader parallelism

    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    save_strategy="epoch",  # Save models only at the end of each epoch for efficiency
    eval_strategy="no",  # Disable intermediate evaluation unless necessary
    logging_steps=500,  # Reduce logging frequency
    save_total_limit=1,  # Save only the most recent checkpoint
    run_name="train",  # W&B run name
)

# 6. Create a trainer
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

# 7. Start training
trainer.train()

# 8. Save the final model
student_model.save_pretrained(config["sentence_embedding_model_save_path"]+"/final_model")

# 9. Load the test dataset
config["data_type"] = "test"
sent_list = get_sent_list(config)
the_original_dataset: Dataset = original_dataset(sent_list)
the_LLM_dataset: Dataset = LLM_dataset(the_original_dataset, config)

test_dataset: DatasetDict = the_LLM_dataset.convert_to_dataset_dict()

teacher_model = SentenceTransformer(args.teacher_model)
# 9. Evaluate the model
from sentence_transformers.evaluation import MSEEvaluator
mse_evaluator = MSEEvaluator(
    source_sentences=test_dataset["output_LLM"],
    target_sentences=test_dataset["input_LLM"],
    teacher_model=teacher_model,
    name="stsb-dev",
)
results = mse_evaluator(student_model)
print(results)
wandb.log({f"results[{mse_evaluator.primary_metric}]":results[mse_evaluator.primary_metric]})
wandb.log({"results":results})