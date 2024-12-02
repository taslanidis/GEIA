import os
from tqdm import tqdm
import copy
import transformers

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine

import json
import numpy as np
import pandas as pd
import argparse
import sys

from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,T5Config,
    AdamW, get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
from attacker_models import SequenceCrossEntropyLoss
from simcse_persona import get_persona_dict
from attacker_evaluation_gpt import eval_on_batch
from datasets import load_dataset
from data_process import get_sent_list

# Sentence Model
class Sentence_Embedding_model(nn.Module):
    def __init__(self, embedding_dim: int = 718, sequence_length:int = 10, model_type:str ="mean"):
        super(Sentence_Embedding_model, self).__init__()

        if model_type == "mean":
            self.model = MeanMapping()
        elif model_type == "linear":
            self.model = LinearMapping(sequence_length)
        elif model_type == "convolution":
            self.model = Conv1DMapping(embedding_dim, sequence_length)
        elif model_type == "self-attention":
            self.model = MultiheadAttentionMapping(embedding_dim, num_heads=4)
        elif model_type == "encoder":
            raise Exception("Not implemented")
        else:
            raise Exception("Not understand")
    
    def forward(self,x):
        return self.model(x) 

class MeanMapping(nn.Module):
    def __init__(self):
        super(MeanMapping, self).__init__()

    def forward(self, x):
        return torch.mean(x,dim=1)

class LinearMapping(nn.Module):
    def __init__(self, sequence_length):
        super(LinearMapping, self).__init__()
        self.fc1 = nn.Linear(sequence_length, 1)

    def forward(self, x):
        batch,length,embedding = x.shape
        x_permuted = x.permute(0,2,1)
        out = self.fc1(x_permuted)
        out = out.squeeze()
        out = out.reshape(batch,embedding)
        return out

class Conv1DMapping(nn.Module):
    def __init__(self, embedding_dim, sequence_length):
        super(Conv1DMapping, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=sequence_length, 
                               out_channels=sequence_length // 2, 
                               kernel_size=3, 
                               padding=1)  # Reduce sequence length by half
        self.conv2 = nn.Conv1d(in_channels=sequence_length // 2, 
                               out_channels=1, 
                               kernel_size=3, 
                               padding=1)  # Collapse to 1 channel

    def forward(self, x):
        # x is of shape [batch, length, embedding]
        batch,length,embedding = x.shape
        x = self.conv1(x)  # Shape: [batch, length / 2, embedding]
        x = torch.relu(x)
        x = self.conv2(x)  # Shape: [batch, 1, embedding]
        x = x.reshape(batch,embedding)  # Shape: [batch, embedding]
        return x

class MultiheadAttentionMapping(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiheadAttentionMapping, self).__init__()
        # MultiheadAttention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            sentence_embedding: Tensor of shape (batch_size, embedding_dim)
        """
        # Use the input as both query, key, and value for self-attention
        # self.multihead_attention expects (batch_size, sequence_length, embedding_dim)
        attended_output, _ = self.multihead_attention(x, x, x)
        
        # Aggregate sequence features to create a sentence embedding
        # Here, using mean pooling. Other methods (e.g., max pooling, weighted sum) can also be used.
        sentence_embedding = attended_output.mean(dim=1)  # (batch_size, embedding_dim)

        return sentence_embedding
    
class linear_projection(nn.Module):
    
    def __init__(self, in_num, out_num=1024):
        super(linear_projection, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, x, use_final_hidden_only = True):
        # x should be of shape (?,in_num) according to gpt2 output
        out_shape = x.size()[-1]
        assert(x.size()[1] == out_shape)
        out = self.fc1(x)
        return out


def init_gpt2():
    config = GPT2Config.from_pretrained('microsoft/DialoGPT-medium')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    model = GPT2LMHeadModel(config)
    return model, tokenizer


def init_opt():
    config = T5Config.from_pretrained('google/t5-large-lm-adapt')
    model = T5ForConditionalGeneration(config)
    tokenizer = T5Tokenizer.from_pretrained("google/t5-large-lm-adapt")
    return model, tokenizer


class the_dataset(Dataset):
    
    def __init__(self, data, model_tokenizer):
        self.data = data
        self.model_tokenizer = model_tokenizer 
        self.model_tokenizer.pad_token = self.model_tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        return  text
        
    def collate(self, unpacked_data):
        # truncation=True is needed but does nothing!
        return unpacked_data, self.model_tokenizer(unpacked_data,return_tensors="pt", padding=True) 


def process_data(
        data, 
        batch_size: int, 
        device1: torch.device, 
        device2: torch.device, 
        config: dict, 
        attacker_emb_size: int
    ):
    model = AutoModelForCausalLM.from_pretrained(config['embed_model_path'], device_map=device1)   # dim 768
    tokenizer = AutoTokenizer.from_pretrained(config['embed_model_path'], padding_side='left')
    dataset = the_dataset(data, tokenizer)
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)

    embedding_dimension: int = 768
    if config['embed_model_path'].find("Mistral") != -1:
        embedding_dimension = 4096
    elif config['embed_model_path'].find("meta") != -1:
        embedding_dimension = 4096

    # Sentence model - In the attacker schema
    sentence_embedding_reduction = Sentence_Embedding_model(embedding_dim=embedding_dimension, sequence_length=config["max_new_tokens"]-1, model_type=config["sentence_aggregation"]).to(device2)
    
    # projection needed if sizes are different
    need_proj: bool = attacker_emb_size != embedding_dimension

    # Extra Projection; aligning model f (victim) with the attacker
    if need_proj:
        projection = linear_projection(in_num=embedding_dimension, out_num=attacker_emb_size).to(device2)
    
    # Attacker Model
    if config['model_dir'] == 'random_gpt2_medium':
        model_attacker, tokenizer_attacker = init_gpt2()
    else:
        model_attacker = AutoModelForCausalLM.from_pretrained(config['model_dir'])
        tokenizer_attacker = AutoTokenizer.from_pretrained(config['model_dir'])
    
    criterion = SequenceCrossEntropyLoss()
    model_attacker.to(device2)
    param_optimizer = list(model_attacker.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_gradients_accumulation = 1
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_train_optimization_steps  = len(dataloader) * num_epochs // num_gradients_accumulation
    optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)
    if need_proj:
        optimizer.add_param_group({'params': projection.parameters()})
        
    optimizer.add_param_group({'params': sentence_embedding_reduction.parameters()})
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100, 
                                            num_training_steps = num_train_optimization_steps)
    
    # 1. Get the embeddings from victim on text
    # 2. Pass through the projection (if embedding spaces don't match)
    # 3. Train the attacker, by:
    #     3.a Taking the sentence embedding and generating text
    #     3.b Calculate the loss compared of the generated text, and the GT text
    print_first_decode = True
    count = 0 
    for i in range(num_epochs):
        model.eval()
        for idx, (batch_text,batch_tokenized_text) in enumerate(tqdm(dataloader, desc="Processing Batches")):
            with torch.no_grad():       
                batch_tokenized_text = batch_tokenized_text.to(device1)
                outputs=model.generate(**batch_tokenized_text, do_sample=True, max_new_tokens=config["max_new_tokens"], top_k=50, output_hidden_states=True, return_dict_in_generate = True, pad_token_id=tokenizer.eos_token_id)

                # get all the tokens expect the tokens that are the input text (i.e. the first token)
                last_hidden_state = torch.cat([ i[-1] for i in outputs["hidden_states"]][1:], dim =1)

                if print_first_decode:
                    print("Example of LLM inference:")
                    print("output:",tokenizer.decode(outputs["sequences"][0][:batch_tokenized_text.input_ids.shape[1]])," --- ",tokenizer.decode(outputs["sequences"][0][batch_tokenized_text.input_ids.shape[1]:],skip_special_tokens=True))
                    print("last_hidden_state shape",last_hidden_state.shape)
                    print_first_decode=False
                
            # Aggregate everything to a single sentence
            last_hidden_state = last_hidden_state.to(device2)
            embeddings =  sentence_embedding_reduction(last_hidden_state)  

            # attacker part, needs training
            if need_proj:
               embeddings = projection(embeddings)

            record_loss, perplexity = train_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=model_attacker,
                tokenizer=tokenizer_attacker,
                criterion=criterion,
                device=device2,
                train=True
            )
            optimizer.step()
            scheduler.step()
            # make sure no grad for GPT optimizer
            optimizer.zero_grad()
            print(f'Training: epoch {i} batch {idx} with loss: {record_loss} and PPL {perplexity} with size {embeddings.size()}')
            wandb.log({"record_loss": record_loss, "perplexity": perplexity})
            if count ==1417:
                break
            count+=1
        if need_proj:
            torch.save(projection.state_dict(), config['projection_save_path'])
        torch.save(sentence_embedding_reduction.state_dict(), config['sentence_embedding_reduction_save_path'])
        model_attacker.save_pretrained(config['attacker_save_path'])
    exit()

### used for testing only
def process_data_test(
        data,
        batch_size: int,
        device: torch.device,
        config: dict,
        attacker_emb_size: int
    ):

    model = AutoModelForCausalLM.from_pretrained(config['embed_model_path'], device_map="auto",)   # dim 768
    base_model = model.base_model
    lm_head = model.lm_head
    tokenizer = AutoTokenizer.from_pretrained(config['embed_model_path'], padding_side="left")
    
    if(config['decode'] == 'beam'):
        save_path = "logs/" + config['attacker_save_name'] + '_beam.log'
    else:
        save_path = "logs/" + config['attacker_save_name'] + '.log'
    
    dataset = the_dataset(data, tokenizer, device)
    # no shuffle for testing data
    dataloader = DataLoader(dataset=dataset, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=dataset.collate)
    print('load data done')

    embedding_dimension: int = 768
    if config['embed_model_path'].find("Mistral") != -1:
        embedding_dimension = 4096
    elif config['embed_model_path'].find("meta") != -1:
        embedding_dimension = 4096


    sentence_embedding_reduction = Sentence_Embedding_model(embedding_dim=embedding_dimension, sequence_length=dataset.max(), model_type=config["sentence_aggregation"])
    sentence_embedding_reduction.load_state_dict(torch.load(config["sentence_embedding_reduction_save_path"]))
    sentence_embedding_reduction.to(device2)

    # projection needed if sizes are different
    need_proj: bool = attacker_emb_size != embedding_dimension

    if need_proj:
        projection = linear_projection(in_num=embedding_dimension, out_num=attacker_emb_size)
        projection.load_state_dict(torch.load(config['projection_save_path']))
        projection.to(device)
        print('load projection done')
    else:
        print('no projection loaded')
    
    # setup on config for sentence generation   AutoModelForCausalLM
    config['model'] = AutoModelForCausalLM.from_pretrained(config['attacker_save_path']).to(device)
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')

    sent_dict = {}
    sent_dict['gt'] = []
    sent_dict['pred'] = []
    with torch.no_grad():  
        count=0
        for idx, (batch_text,batch_tokenized_text) in enumerate(tqdm(dataloader, desc="Processing Batches")):
            print("batch_text:", batch_text)
            print("batch_tokenized_text",batch_tokenized_text.keys())
            print("decoded batch_text:",tokenizer.decode(batch_tokenized_text["input_ids"][0],skip_special_tokens=True))
            output = base_model(**batch_tokenized_text)
                
            hidden_state = output.last_hidden_state

            output = lm_head(hidden_state)
            print("DECODED:")
            print(tokenizer.decode(torch.argmax(output[0],-1),skip_special_tokens=True))
            # Aggregate everything to a single sentence
            print("hidden_state.shape",hidden_state.shape)
            embeddings =  sentence_embedding_reduction(hidden_state)
            print("embeddings.shape",embeddings.shape)
            if need_proj:
                embeddings = projection(embeddings)

            sent_list, gt_list = eval_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=config['model'],
                tokenizer=config['tokenizer'],
                device=device,
                config=config
            )    
            # print(f'Testing {idx} batch done with {idx*batch_size} samples')
            sent_dict['pred'].extend(sent_list)
            sent_dict['gt'].extend(gt_list)

            print("gt_list", gt_list)
            print("sent_list", sent_list)
            if count==5:
                exit()
            count+=1
            print("---------")
        with open(save_path, 'w') as f:
            json.dump(sent_dict, f, indent=4)

    return 0
        


def train_on_batch(
        batch_X: torch.TensorType,
        batch_D: torch.TensorType,
        model,
        tokenizer,
        criterion,
        device: torch.DeviceObjType,
        train: bool = True
    ):
    """
    batch_X: output from model f (victim model) with the sentence embeddings
    batch_D: the text that the attacker needs to generate, in order to have a perfect inversion
    """
    # padding_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize the text (gt) needed for a perfect inversion
    inputs = tokenizer(batch_D, return_tensors='pt', padding='max_length', truncation=True, max_length=40)
    #dial_tokens = [tokenizer.encode(item) + turn_ending for item in batch_D]
    #print(inputs)
    input_ids = inputs['input_ids'].to(device) # tensors of input ids
    labels = input_ids.clone()
    #print(input_ids.size())

    # embed the input ids of the gt text using GPT-2 embedding
    input_emb = model.transformer.wte(input_ids)
    
    # add extra dim to cat together
    batch_X = batch_X.to(device)
    batch_X_unsqueeze = torch.unsqueeze(batch_X, 1)
    
    # The first token becomes the sentence embedding
    # Rest follows the gt text [Only for training]
    # So the labels are leading always 1 to the ground truth as seen in Figure 2 of the paper
    inputs_embeds = torch.cat((batch_X_unsqueeze, input_emb),dim=1) # [batch, 1 + max_length, emb_dim]
    # print(inputs_embeds.size())
    past = None
    # need to move to device later
    inputs_embeds = inputs_embeds

    #logits, past = model(inputs_embeds=inputs_embeds,past = past)
    logits, past = model(inputs_embeds=inputs_embeds, past_key_values=past, return_dict=False)
    logits = logits[:, :-1].contiguous()
    target = labels.contiguous()
    target_mask = torch.ones_like(target).float()
    loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")   

    record_loss = loss.item()
    perplexity = np.exp(record_loss)
    if train:
        loss.backward()

    return record_loss, perplexity



if __name__ == '__main__':
    model_cards = {}
    model_cards['mistralai']='mistralai/Mistral-7B-v0.1'
    model_cards["t5-base"]= 'google-t5/t5-base'
    model_cards["meta-llama"] = "meta-llama/Meta-Llama-3-8B"
    
    parser = argparse.ArgumentParser(description='Training external decoder to attack an LLM')
    parser.add_argument('--model_dir', type=str, default='microsoft/DialoGPT-large', help='Dir of your model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epoches.')
    parser.add_argument('--max_new_tokens', type=int, default=60, help='How many tokens should the LLM use to answer the prompt?')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch_size #.')
    parser.add_argument('--dataset', type=str, default='personachat', help='Name of dataset: personachat or qnli')
    parser.add_argument('--data_type', type=str, default='test', help='train/test')
    parser.add_argument('--embed_model', type=str, default='meta-llama', help='Name of embedding model: mistralai/...')
    parser.add_argument('--decode', type=str, default='beam', help='Name of decoding methods: beam/sampling')
    parser.add_argument('--sentence_aggregation', type=str, default='mean', help='Name of sentence embending creation methods: mean/linear/convolution/self-attention/encoder')
    args = parser.parse_args()
    print("ARGS:")
    print(args)
    
    config = {}
    config['model_dir'] = args.model_dir
    config['num_epochs'] = args.num_epochs
    config['batch_size'] = args.batch_size
    config['dataset'] = args.dataset
    config['data_type'] = args.data_type
    config['embed_model'] = args.embed_model
    config['decode'] = args.decode
    config['embed_model_path'] = model_cards[config['embed_model']]
    config["sentence_aggregation"] = args.sentence_aggregation
    config["max_new_tokens"] = args.max_new_tokens

    # custom save paths
    attacker_model: str = "rand_gpt2_m" if args.model_dir == "random_gpt2_medium" else "dialogpt2"
    attacker_save_name: str = f"attacker_{attacker_model}_{config['dataset']}_{config['embed_model']}_{config['sentence_aggregation']}"
    projection_save_name: str = f"projection_{attacker_model}_{config['dataset']}_{config['embed_model']}_{config['sentence_aggregation']}"
    sentence_embedding_reduction_save_name: str = f"sentence_embedding_reduction_{attacker_model}_{config['dataset']}_{config['embed_model']}_{config['sentence_aggregation']}"
    config['attacker_save_path'] = f"models/{attacker_save_name}"
    config['projection_save_path'] = f"models/{projection_save_name}"
    config['sentence_embedding_reduction_save_path'] = f"models/{sentence_embedding_reduction_save_name}"
    config['attacker_save_name'] = attacker_save_name
    config['projection_save_name'] = projection_save_name

    if torch.cuda.is_available():
        config['device1'] = torch.device("cuda:0")
        if torch.cuda.device_count()>1:
            config['device2'] = torch.device("cuda:1")
        else:
            print("Only 1 cuda device was found!")
            config['device2'] = torch.device("cuda:0")
        print("device 1 :",config["device1"])
        print("device 2 :",config["device2"])
    else:
        raise ImportError("Torch couldn't find CUDA devices. Can't run the attacker on CPU.")
    
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    config['eos_token'] = config['tokenizer'].eos_token
    config['use_opt'] = False

    device1 = config['device1']
    device2 = config['device2']
    batch_size = config['batch_size']

    wandb.init(project="GEIA", name= "attack_to_LLM", config = config)

    sent_list = get_sent_list(config)
    
    # TODO: figure out when projection is necessary
    attacker_emb_size = 768
    if config['model_dir'].find("medium") != -1:
        attacker_emb_size = 1024
    elif config['model_dir'].find("large") != -1:
        attacker_emb_size = 1280


    if(config['data_type'] == 'train'):
        # -- Training --
        if('simcse' in config['embed_model']):
            process_data_simcse(sent_list,batch_size,device1,device2,config,attacker_emb_size)
        else:
            process_data(sent_list,batch_size,device1,device2,config,attacker_emb_size)

    elif(config['data_type'] == 'test'):
        # -- Inference --
        if('simcse' in config['embed_model']):
            process_data_test_simcse(sent_list,batch_size,device1,device2,config,attacker_emb_size=attacker_emb_size)
        else:
            process_data_test(sent_list,batch_size,device1,device2,config,attacker_emb_size=attacker_emb_size)

    wandb.finish()
