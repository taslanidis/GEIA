from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.training_args import BatchSamplers

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import argparse
import wandb
from data_process import get_sent_list
from dataset import original_dataset, LLM_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument(
    "--dataset",
    type=str,
    default="fingpt-sentiment",
    help="Name of dataset: personachat or qnli or fingpt-sentiment",
)
parser.add_argument("--student_model", type=str, default="all-roberta-large-v1")  
parser.add_argument("--teacher_model", type=str, default="all-roberta-large-v1")
parser.add_argument("--model_save_path", type=str, default="models")
parser.add_argument("--seed", type=int, default=82)
parser.add_argument(
        "--save_embedding",
        type=str,
        default="./LLM_hidden_states2/",
        help="The path to save the LLM last hidden states",
    )
parser.add_argument("--embed_model_path", type=str, default = "meta-llama/Llama-2-7b-chat-hf")
parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="How many tokens should the LLM use to answer the prompt?",
    )
parser.add_argument(
    "--percentage_of_dataset",
    type=int,
    default=1,
    help="how much of the dataset do you want to upload",
)
args = parser.parse_args()


config = {}
config["max_new_tokens"] = args.max_new_tokens
config["dataset"] = args.dataset
config["student_model"] = args.student_model
config["teacher_model"] = args.teacher_model
config["batch_size"] = args.batch_size
config["num_epochs"] = args.num_epochs
config["data_type"] = "train"
config["seed"] = args.seed
config["save_embedding"] = args.save_embedding
config["embed_model_path"] = args.embed_model_path
config["model_save_path"] = args.model_save_path + "/" + args.student_model + "_student_" + args.teacher_model + "_teacher" + args.dataset + "_dataset"
config["last_hidden_states_flag"] = False
if torch.cuda.is_available():
    config["device"] = torch.device("cuda:0")
else:
    raise ImportError(
        "Torch couldn't find CUDA devices. Can't run the attacker on CPU."
    )


print("args:", args)
# wandb.init(project="GEIA", name = "sentence_encoder", config=config)

# Load the dataset
sent_list = get_sent_list(config)
the_original_dataset: Dataset = original_dataset(sent_list)
# reduce the original_dataset to just args.percentage_of_dataset samples to check the code
the_original_dataset = torch.utils.data.Subset(the_original_dataset, range(np.floor(len(the_original_dataset) * args.percentage_of_dataset/100).astype(int)))
teacher_model = SentenceTransformer(args.teacher_model)
the_LLM_dataset: Dataset = LLM_dataset(the_original_dataset, config, teacher_model)

the_LLM_dataset[0]

# Load the model
student_model = SentenceTransformer(args.student_model)

# Define the train loss
train_loss = losses.MSELoss(model=student_model)

# 5. Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=args.model_save_path,
    # Optional training parameters:
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="train",  # Will be used in W&B if `wandb` is installed
)

# 6. Create a trainer
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    train_loss=train_loss,
)

# 7. Start training
trainer.train()

# 8. Save the final model
student_model.save_pretrained(args.model_save_path+"/final_model")

# 9. Load the test dataset
config["data_type"] = "test"
sent_list = get_sent_list(config)
the_original_dataset: Dataset = original_dataset(sent_list)
the_LLM_dataset: Dataset = LLM_dataset(the_original_dataset, config)

# 9. Evaluate the model
from sentence_transformers.evaluation import MSEEvaluator
mse_evaluator = MSEEvaluator(
    source_sentences=the_LLM_dataset,
    target_sentences=the_LLM_dataset,
    teacher_model=teacher_model,
    name="test",
)
results = mse_evaluator(student_model)
wandb.log(results[mse_evaluator.primary_metric], step=trainer.global_step)
wandb.log(results, step=trainer.global_step)