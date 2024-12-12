import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataset import original_dataset, LLM_dataset
import wandb

import torch
import torch.nn as nn

import json
import numpy as np
import pandas as pd
import argparse
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from attacker_models import SequenceCrossEntropyLoss
from attacker_evaluation_gpt import eval_on_batch_fast
from data_process import get_sent_list
from sentence_representation_model import Sentence_Embedding_model

class linear_projection(nn.Module):

    def __init__(self, in_num, out_num=1024):
        super(linear_projection, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, x, use_final_hidden_only=True):
        # x should be of shape (?,in_num) according to gpt2 output
        out_shape = x.size()[-1]
        assert x.size()[1] == out_shape
        out = self.fc1(x)
        return out


def init_gpt2():
    config = GPT2Config.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def process_data(
    dataset: LLM_dataset,
    config: dict,
):
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=dataset.collate,
    )

    # Sentence model - In the attacker schema
    sentence_embedding_model = Sentence_Embedding_model(
        model_type=config["sentence_aggregation"], device=config["device"]
    ).to(config["device"])
    sentence_embedding_model.eval()

    # projection needed if sizes are different
    need_proj: bool = config["attacker_emb_size"] != config["victim_emb_size"]

    # Extra Projection; aligning model f (victim) with the attacker
    if need_proj:
        projection = linear_projection(
            in_num=config["victim_emb_size"], out_num=config["attacker_emb_size"]
        ).to(config["device"])

    # Attacker Model
    if config["model_dir"] == "random_gpt2_medium":
        model_attacker, tokenizer_attacker = init_gpt2()
    else:
        model_attacker = AutoModelForCausalLM.from_pretrained(config["model_dir"])
        tokenizer_attacker = AutoTokenizer.from_pretrained(config["model_dir"])

    criterion = SequenceCrossEntropyLoss()
    model_attacker.to(config["device"])
    param_optimizer = list(model_attacker.named_parameters())
    no_decay = ["bias", "ln", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_gradients_accumulation = 1
    num_epochs = config["num_epochs"]
    num_train_optimization_steps = (
        len(dataloader) * num_epochs // num_gradients_accumulation
    )
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-06)
    if need_proj:
        optimizer.add_param_group({"params": projection.parameters()})


    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps
    )

    # 1. Get the embeddings from victim on text
    # 2. Pass through the projection (if embedding spaces don't match)
    # 3. Train the attacker, by:
    #     3.a Taking the sentence embedding and generating text
    #     3.b Calculate the loss compared of the generated text, and the GT text
    print_first_decode = True
    count = 0
    for epoch in range(num_epochs):
        for idx, (batch_text, LLM_output) in enumerate(
            tqdm(dataloader, desc="Processing Batches")
        ):
            with torch.no_grad():
                embeddings = sentence_embedding_model.model.encode(LLM_output,convert_to_tensor=True).to(config["device"])

            # attacker part, needs training
            if need_proj:
                embeddings = projection(embeddings)

            record_loss, perplexity = train_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=model_attacker,
                tokenizer=tokenizer_attacker,
                criterion=criterion,
                device=config["device"],
                train=True,
            )

            optimizer.step()
            scheduler.step()
            # make sure no grad for GPT optimizer
            optimizer.zero_grad()

            # print(
            #     f"Training: epoch {i} batch {idx} with loss: {record_loss} and PPL {perplexity} with size {embeddings.size()}"
            # )

            wandb.log({"record_loss": record_loss, "perplexity": perplexity, "epoch": epoch})

        if need_proj:
            torch.save(projection.state_dict(), config["projection_save_path"])
        model_attacker.save_pretrained(config["attacker_save_path"])


### used for testing only
def process_data_test(dataset: LLM_dataset, config: dict):

    if config["decode"] == "beam":
        save_path = "logs/" + config["attacker_save_name"] + "_beam.log"
    else:
        save_path = "logs/" + config["attacker_save_name"] + ".log"

    # no shuffle for testing data
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=dataset.collate,
    )
    print("load data done")

    sentence_embedding_model = Sentence_Embedding_model(
        model_type=config["sentence_aggregation"],
    )
    sentence_embedding_model.eval()

    # projection needed if sizes are different
    need_proj: bool = config["attacker_emb_size"] != config["victim_emb_size"]

    if need_proj:
        projection = linear_projection(
            in_num=config["victim_emb_size"], out_num=config["attacker_emb_size"]
        )
        projection.load_state_dict(torch.load(config["projection_save_path"]))
        projection.to(config["device"])
        projection.eval()
        print("load projection done")
    else:
        print("no projection loaded")

    # setup on config for sentence generation   AutoModelForCausalLM
    config["model"] = AutoModelForCausalLM.from_pretrained(
        config["attacker_save_path"]
    ).to(config["device"])
    config["model"].eval()
    config["tokenizer"] = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

    sent_dict = {}
    sent_dict["gt"] = []
    sent_dict["pred"] = []
    with torch.no_grad():
        count = 0
        for idx, (batch_text, LLM_output) in enumerate(
            tqdm(dataloader, desc="Processing Batches")
        ):
            
            embeddings = sentence_embedding_model.model.encode(LLM_output,convert_to_tensor=True).to(config["device"])

            if need_proj:
                embeddings = projection(embeddings)

            sent_list, gt_list = eval_on_batch_fast(
                batch_X=embeddings,
                batch_D=batch_text,
                model=config["model"],
                tokenizer=config["tokenizer"],
                device=config["device"],
                config=config,
            )

            # print(f'Testing {idx} batch done with {idx*batch_size} samples')
            sent_dict["pred"].extend(sent_list)
            sent_dict["gt"].extend(gt_list)

        with open(save_path, "w") as f:
            json.dump(sent_dict, f, indent=4)


def train_on_batch(
    batch_X: torch.TensorType,
    batch_D: torch.TensorType,
    model,
    tokenizer,
    criterion,
    device: torch.DeviceObjType,
    train: bool = True,
):
    """
    batch_X: output from model f (victim model) with the sentence embeddings
    batch_D: the text that the attacker needs to generate, in order to have a perfect inversion
    """
    # padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize the text (gt) needed for a perfect inversion
    inputs = tokenizer(
        batch_D,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=40,
    )
    # dial_tokens = [tokenizer.encode(item) + turn_ending for item in batch_D]
    # print(inputs)
    input_ids = inputs["input_ids"].to(device)  # tensors of input ids
    labels = input_ids.clone()
    # print(input_ids.size())

    # embed the input ids of the gt text using GPT-2 embedding
    input_emb = model.transformer.wte(input_ids)

    # add extra dim to cat together
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)

    # The first token becomes the sentence embedding
    # Rest follows the gt text [Only for training]
    # So the labels are leading always 1 to the ground truth as seen in Figure 2 of the paper
    inputs_embeds = torch.cat(
        (batch_X_unsqueeze, input_emb), dim=1
    )  # [batch, 1 + max_length, emb_dim]
    # print(inputs_embeds.size())
    past = None
    # need to move to device later
    inputs_embeds = inputs_embeds

    # logits, past = model(inputs_embeds=inputs_embeds,past = past)
    logits, past = model(
        inputs_embeds=inputs_embeds, past_key_values=past, return_dict=False
    )
    logits = logits[:, :-1].contiguous()
    target = labels.contiguous()
    target_mask = torch.ones_like(target).float()
    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")

    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    if train:
        loss.backward()

    return record_loss, perplexity


if __name__ == "__main__":
    model_cards = {}
    model_cards["mistralai"] = "mistralai/Mistral-7B-v0.1"
    model_cards["t5-base"] = "google-t5/t5-base"
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch_size #.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="personachat",
        help="Name of dataset: personachat or qnli or fingpt-sentiment",
    )
    parser.add_argument("--data_type", type=str, default="test", help="train/test")
    parser.add_argument(
        "--embed_model",
        type=str,
        default="meta-llama",
        help="Name of embedding model: mistralai/...",
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
        help="Name of sentence embending creation methods: mean/linear/convolution/self-attention/encoder",
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
        default=10,
        help="how much of the dataset do you want to upload",
    )
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
    config["data_type"] = args.data_type
    config["embed_model"] = args.embed_model
    config["decode"] = args.decode
    config["embed_model_path"] = model_cards[config["embed_model"]]
    config["sentence_aggregation"] = args.sentence_aggregation
    config["max_new_tokens"] = args.max_new_tokens
    config["seed"] = args.seed
    config['use_opt'] = False
    config["last_hidden_states_flag"] = False

    # custom save paths
    attacker_model: str = (
        "rand_gpt2_m" if args.model_dir == "random_gpt2_medium" else "dialogpt2"
    )

    config[
        "save_embedding"
    ]: str = f"{args.save_embedding}/saved_embeddings_{config['dataset']}_{config['embed_model']}_{config['max_new_tokens']}_{config['data_type']}_{args.percentage_of_dataset}"

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

    if torch.cuda.is_available():
        config["device"] = torch.device("cuda:0")
    else:
        raise ImportError(
            "Torch couldn't find CUDA devices. Can't run the attacker on CPU."
        )

    config["victim_emb_size"]: int = 768
    if config['sentence_aggregation'] == 'all-roberta-large-v1':
        config["victim_emb_size"] = 1024
    elif config['sentence_aggregation'] == 'sent_t5_large':
        config["victim_emb_size"] = 1024
    elif config['sentence_aggregation'].find('simcse') != -1:
        config["victim_emb_size"] = 1024

    config["attacker_emb_size"]: int = 768
    if config["model_dir"].find("medium") != -1:
        config["attacker_emb_size"] = 1024
    elif config["model_dir"].find("large") != -1:
        config["attacker_emb_size"] = 1280

    print("Configuration:")
    print(config)

    wandb.init(project="GEIA", name="attack_to_LLM", config=config)

    sent_list = get_sent_list(config)
    the_original_dataset: Dataset = original_dataset(sent_list)
    # reduce the original_dataset to just args.percentage_of_dataset samples to check the code
    the_original_dataset = torch.utils.data.Subset(the_original_dataset, range(np.floor(len(the_original_dataset) * args.percentage_of_dataset/100).astype(int)))
    the_LLM_dataset: Dataset = LLM_dataset(the_original_dataset, config)

    if config["data_type"] == "train":
        # -- Training --
        process_data(the_LLM_dataset, config)

    elif config["data_type"] == "test":
        # -- Inference --
        process_data_test(the_LLM_dataset, config)

    wandb.finish()
