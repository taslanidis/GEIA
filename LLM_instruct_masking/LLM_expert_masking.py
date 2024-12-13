import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from datasets import load_dataset  # Import the datasets library
from tqdm import tqdm  # Import tqdm for progress visualization
from torch.utils.data import DataLoader  # Import DataLoader for batching

SCRATCH_DIR = os.path.join("/scratch-shared", os.getenv("USER"))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU number. If inference with multiple GPUs, set multiple GPU numbers
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the system prompt for text-to-video (t2v) conversion
sys_prompt = """
You are part of a team of bots tasked with obfuscating sensitive information in text while preserving its overall structure and meaning. Your role is to identify sensitive tokens, such as named entities (e.g., names, locations, and organizations), and replace them with appropriate placeholders. The approach should mirror the techniques used in redacting classified CIA documents, ensuring clarity and security.

Rules:
1. **Sensitive Entities**: These include names, locations, and other identifiable information that could compromise privacy.
2. **Masking Format**: Replace sensitive entities with generic placeholders, such as `<NAME>`, `<LOCATION>`, `<YEAR>`, `<ENTITY>`, etc., depending on the context.
3. **Preservation of Context**: Ensure that the resulting text is still grammatically correct and retains its original meaning as much as possible.
4. **Consistency**: Use the same placeholder for identical sensitive entities within a single sequence.
5. **Edge Cases**: If no sensitive entities are found, output the sequence unchanged.

Respond only with the processed sequences when handling user inputs.
"""

def mask_passage_batch(prompts: list, model, tokenizer, do_sample: bool, max_length: int, temperature: float = 0.7, top_p: float = 0.9):
    # Prepare batched inputs
    messages = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f'Please identify and mask all sensitive entities in the following passage: "{prompt}". Please leave only one less informative/sensitive information related to an entity unmasked.'}
        ] for prompt in prompts
    ]
    
    # Vectorized tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,  # Avoid returning PyTorch tensors for compatibility
        return_dict=True
    )
    inputs_list = inputs.to(device)  # Move the entire batch to device

    # Pad the inputs to ensure consistent dimensions
    padded_inputs = tokenizer.pad(
        inputs_list,
        padding=True,  # Add padding to the shorter sequences
        return_tensors="pt"  # Convert the padded sequences to PyTorch tensors
    ).to(device)

    # Set generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "do_sample": do_sample,
        "top_k": 1,
    }
    with torch.no_grad():
        outputs = model.generate(**padded_inputs, **gen_kwargs)
        outputs = outputs[:, padded_inputs['input_ids'].shape[1] + 1:]
        generated_txts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(generated_txts[0])
    return generated_txts



if __name__ == "__main__":
    # Argument parser for user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='sentence-transformers/altlex', help="Name of the dataset to use")
    parser.add_argument("--model_path", type=str, default='GLM-4/glm-4-9b-chat', help="Path to the pre-trained Llama-3 model")
    parser.add_argument("--max_length", type=int, default=2500, help="Maximum length of the generated description")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--log_file", type=str, default='output.json', help="Path to the log file for entries")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)  # Load the altlex dataset

    # Load the model and tokenizer
    model_path = os.path.join(SCRATCH_DIR, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    # Create a DataLoader for batching
    batch_size = 128  # Adjust batch size as needed
    data_loader = DataLoader(
    dataset['train'], 
    batch_size=batch_size, 
    shuffle=False, 
    pin_memory=True, 
    num_workers=4  # Adjust based on your system
    )

    # Open the log file for writing outside the loop
    with open(args.log_file, 'w', encoding='utf-8') as f:
        # Iterate over the dataset in batches
        for batch in tqdm(data_loader, desc="Processing batches"):
            prompts = [sentence for sentence in batch['simplified']]  # Get the original texts
            masked_txts = mask_passage_batch(prompts, model, tokenizer, do_sample=True, max_length=args.max_length, temperature=args.temperature, top_p=args.top_p)

            # Log the original and generated texts in JSON format
            for original_txt, masked_txt in zip(batch['simplified'], masked_txts):
                log_entry = {
                    "original": original_txt,
                    "masked": masked_txt
                }
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')  # Write a newline for each entry to separate them

#example usage
# python LLM_expert_masking.py