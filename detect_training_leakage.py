import argparse
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,T5Config,
    AdamW, get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer

from attacker import (
    get_sent_list, 
    linear_projection
)


class ExtensionData(Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]
        return  text
        
    def collate(self, unpacked_data):
        return unpacked_data


def eval_ll(
        sentence_embeddings_from_mask: torch.TensorType,
        positive_sample_text: torch.TensorType,
        negative_sample_text: torch.TensorType,
        model,
        tokenizer,
        device: torch.device,
        config: dict
    ):

    if(not config['use_opt']):
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move embeddings to the device
    sentence_embeddings_from_mask = sentence_embeddings_from_mask.to(device)

    # Tokenize positive and negative texts
    positive_inputs = tokenizer(positive_sample_text, return_tensors="pt", padding=True).to(device)
    negative_inputs = tokenizer(negative_sample_text, return_tensors="pt", padding=True).to(device)

    sentence_embeddings = sentence_embeddings_from_mask.unsqueeze(1).to(device)

    # Compute log-likelihood for positive and negative samples
    def compute_log_likelihood(inputs):
        with torch.no_grad():
            labels = inputs['input_ids']
            input_ids = inputs['input_ids']
            embeddings = model.transformer.wte(input_ids).to(device)
            
            print("Emb size: ", embeddings.size())
            print("Sent from mask: ", sentence_embeddings.size())
            # Concatenate token embeddings with expanded sentence embeddings
            inputs_embeds = torch.cat([embeddings, sentence_embeddings], dim=1)

            outputs = model(
                inputs_embeds=inputs_embeds
            )

            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute log-likelihood
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            sequence_log_prob = token_log_probs.sum(dim=-1)  # Sum over tokens
        return sequence_log_prob

    pos_ll = compute_log_likelihood(positive_inputs)
    neg_ll = compute_log_likelihood(negative_inputs)

    return pos_ll.cpu().tolist(), neg_ll.cpu().tolist()


def calculate_leakage(
        original_sent_list: list,
        masked_sent_list: list,
        similar_sent_list: list,
        batch_size: int,
        device: torch.device,
        config: dict,
        attacker_emb_size: int
    ):
    sentence_model = SentenceTransformer(config['embed_model_path'], device=device)

    if(config['decode'] == 'beam'):
        save_path = "logs/leakage_" + config['attacker_save_name'] + '_beam.log'
    else:
        save_path = "logs/leakage_" + config['attacker_save_name'] + '.log'
    
    original_data = ExtensionData(original_sent_list)
    masked_data = ExtensionData(masked_sent_list)
    similar_data = ExtensionData(similar_sent_list)

    # no shuffle for testing data
    original_dataloader = DataLoader(dataset=original_data, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=original_data.collate)
    masked_dataloader = DataLoader(dataset=masked_data, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=masked_data.collate)
    similar_dataloader = DataLoader(dataset=similar_data, 
                              shuffle=False, 
                              batch_size=batch_size, 
                              collate_fn=similar_data.collate)

    print('load data done')

    embedding_dimension: int = 768
    if config['embed_model_path'] == 'all-roberta-large-v1':
        embedding_dimension = 1024
    elif config['embed_model_path'] == 'sent_t5_large':
        embedding_dimension = 1024
    elif config['embed_model_path'].find('simcse') != -1:
        embedding_dimension = 1024

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

    leakage_dict = {}
    leakage_dict['positive_sample_likelihood'] = []
    leakage_dict['negative_sample_likelihood'] = []

    with torch.no_grad():  
        for batch_original, batch_masked, batch_similar in zip(original_dataloader, masked_dataloader, similar_dataloader):

            # The sentence model is a version of Bert, so the <mask> token is known to the model
            embeddings = sentence_model.encode(batch_masked, convert_to_tensor=True).to(device)
  
            if need_proj:
                embeddings = projection(embeddings)

            pos_ll, neg_ll = eval_ll(
                sentence_embeddings_from_mask=embeddings,
                positive_sample_text=batch_original,
                negative_sample_text=batch_similar,
                model=config['model'],
                tokenizer=config['tokenizer'],
                device=device,
                config=config
            )    
            leakage_dict['positive_sample_likelihood'].extend(pos_ll)
            leakage_dict['negative_sample_likelihood'].extend(neg_ll)
        
        with open(save_path, 'w') as f:
            json.dump(leakage_dict, f, indent=4)

    return 0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training external NN as baselines')
    parser.add_argument('--model_dir', type=str, default='random_gpt2_medium', help='Dir of your model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epoches.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch_size #.')
    parser.add_argument('--dataset_path', type=str, default='./data/test.jsonl', help='Path for the processed SNLI dataset.')
    parser.add_argument('--decode', type=str, default='beam', help='Name of decoding methods: beam/sampling')
    args = parser.parse_args()
    
    config = {}
    config['model_dir'] = args.model_dir
    config['batch_size'] = args.batch_size
    config['dataset_path'] = args.dataset_path
    config['dataset'] = 'extension'
    config['decode'] = args.decode
    # victim model
    config['embed_model'] = 'sent_roberta'
    config['embed_model_path'] = 'all-roberta-large-v1'
    config['pretrained_geia_dataset'] = 'personachat'
    # custom save paths
    attacker_model: str = "rand_gpt2_m" if args.model_dir == "random_gpt2_medium" else "dialogpt2"
    attacker_save_name: str = f"attacker_{attacker_model}_{config['pretrained_geia_dataset']}_{config['embed_model']}"
    projection_save_name: str = f"projection_{attacker_model}_{config['pretrained_geia_dataset']}_{config['embed_model']}"
    config['attacker_save_path'] = f"models/{attacker_save_name}"
    config['projection_save_path'] = f"models/{projection_save_name}"
    config['attacker_save_name'] = attacker_save_name
    config['projection_save_name'] = projection_save_name

    if torch.cuda.is_available():
        config['device'] = torch.device("cuda")
    else:
        raise ImportError("Torch couldn't find CUDA devices. Can't run the attacker on CPU.")
    
    config['tokenizer'] = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
    config['eos_token'] = config['tokenizer'].eos_token
    config['use_opt'] = False

    device = config['device']
    batch_size = config['batch_size']

    # unpacking values
    original_sent_list, masked_sent_list, similar_sent_list = get_sent_list(config)
    
    attacker_emb_size = 768
    if config['model_dir'].find("medium") != -1:
        attacker_emb_size = 1024
    elif config['model_dir'].find("large") != -1:
        attacker_emb_size = 1280

    # Measuring Leakage on Pre-Trained Dataset
    calculate_leakage(
        original_sent_list=original_sent_list,
        masked_sent_list=masked_sent_list,
        similar_sent_list=similar_sent_list,
        batch_size=batch_size,
        device=device,
        config=config,
        attacker_emb_size=attacker_emb_size
    )
