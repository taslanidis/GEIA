import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

class original_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        return text

    def collate(self, unpacked_data):
        # truncation=True is needed but does nothing!
        return unpacked_data


class LLM_dataset(Dataset):
    def __init__(self, dataset: Dataset, config: dict):
        testing = False
        self.save_embedding_path = Path(config["save_embedding"])
        if (
            not self.save_embedding_path.exists()
            or (
                self.save_embedding_path.is_dir()
                and not any(p.is_file() for p in self.save_embedding_path.iterdir())
            )
            or testing
        ):
            print("No embeddings found. Generating embeddings.")
            self.save_embedding_path.mkdir(parents=True, exist_ok=True)
            self.create_dataset(dataset, config)

        self.last_hidden_states_flag = config["last_hidden_states_flag"]

        if self.last_hidden_states_flag:
            tokenizer = AutoTokenizer.from_pretrained(
                config["embed_model_path"], padding_side="left"
            )
            decoded_texts, hidden_states_list = [] , []
            for batch_file in tqdm((self.save_embedding_path/ "hidden_states").iterdir(), desc="Loading all toches"):
                batch_data = torch.load(batch_file)
                
                # Decode the input tensor
                input_tensor = batch_data["input_ids"]
                decoded_text = tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
                
                # Append decoded text and hidden states
                decoded_texts.append(decoded_text)
                hidden_states_list.append(batch_data["hidden_states"])
        
            self.mapping = {}
            flag = True
            for text_batch, hidden_state_batch in tqdm(zip(decoded_texts, hidden_states_list), "ziping text with corresponding last_hidden_state"):
                for text, hidden_state in zip(text_batch, hidden_state_batch):
                    self.mapping[text] = hidden_state
                    if flag:
                        print("\n An example of the input:",text)
                        flag = False
            del tokenizer
            torch.cuda.empty_cache()
        else:
            self.mapping = json.load(open(Path(self.save_embedding_path/"output_text.json")))
            # if "Your task is to" in next(iter(self.mapping.values())):
            #     print("The prompting is in the answer. I should be removed!")
            #     flag=True
            #     for keys in self.mapping:
                    
            #         if flag:
            #             print("Example:")
            #             print("BEFORE:\t",self.mapping[keys])
            #             self.mapping[keys] = (" ").join(self.mapping[keys].split("AI:")[1:]).strip()
            #             print("AFTER:\t",self.mapping[keys])
            #             flag = False

            #         self.mapping[keys] = (" ").join(self.mapping[keys].split("AI:")[1:]).strip()
        self.text = list(self.mapping.keys())

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

    def collate(self, batch_input):
        LLM_output = None
        if self.last_hidden_states_flag:
            LLM_output = [self.mapping[text] for text in batch_input]
            LLM_output = torch.stack(LLM_output)
        else:
            LLM_output = [self.mapping[text] for text in batch_input]
        return batch_input, LLM_output

    def create_dataset(self, dataset: Dataset, config: dict):
        model = AutoModelForCausalLM.from_pretrained(
            config["embed_model_path"], device_map=config["device"]
        )  # dim 768
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            config["embed_model_path"], padding_side="left"
        )

        tokenizer.pad_token = tokenizer.eos_token

        dataset: Dataset = original_dataset(dataset)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=config["batch_size"] 
            # if config["batch_size"] < 32 else 32,
        )

        output_text = {}

        hidden_states = self.save_embedding_path / "hidden_states"

        hidden_states.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            sys_prompt = "The following is a friendly conversation between a human and an AI. The response of the AI is single, thoughtful, enthusiastic and engaging sentence that builds meaningfully on what the human said, adds depth to the conversation, and invites further dialogue with the human. If the AI does not know the answer to a question, it truthfully says it does not know. Your task is to just output the response of the AI not the human as well. Current conversation:"
            epoch = 0
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                flag= False
                print("EPOCH",epoch)
                # for i in batch:
                #     if len(i)>=2000:
                #         print("LEN[i]",len(i))
                #         flag=True
                # if not flag:
                #     epoch+=1
                #     continue

                tokenized_without_prompt = tokenizer(
                    batch, padding=True, return_tensors="pt"
                ).to(config["device"])

                batch_with_prompt = [
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ] for prompt in batch
                ]
                # Prepend the prompt to each item in the batch
                # batch_with_prompt = [f"""The following is a friendly conversation between a human and an AI.
                    # The response of the AI is single, thoughtful, enthusiastic and engaging sentence that builds meaningfully on what the human said, adds depth to the conversation, and invites further dialogue with the human.
                    # If the AI does not know the answer to a question, it truthfully says it does not know.
                    # Your task is to provide a short, concise response as the AI, without including any follow-up from the human.
                    # Current conversation:
                    # Human: {item}
                    # AI:""" for item in batch]

                # Tokenize the batch with the prompt added
                tokenized_batch_with_chat_template = tokenizer.apply_chat_template(
                                batch_with_prompt,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_tensors=None,  # Avoid returning PyTorch tensors for compatibility
                                return_dict=True
                            )

                tokenized_prompt_batch = tokenizer.pad(
                    tokenized_batch_with_chat_template,
                    padding=True,  # Add padding to the shorter sequences
                    return_tensors="pt"  # Convert the padded sequences to PyTorch tensors
                ).to(config["device"])

                output = model.generate(
                    **tokenized_prompt_batch,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.8,
                    # num_beams=2,
                    # length_penalty=0.9,
                    repetition_penalty=1.2,
                    max_new_tokens=config["max_new_tokens"],
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # batch_last_hidden_state = torch.cat(
                #     [i[-1] for i in output["hidden_states"]][1:], dim=1
                # )

                # batch_data = {
                #     "input_ids": tokenized_without_prompt.input_ids,
                #     "hidden_states": batch_last_hidden_state
                # }
                # torch.save(batch_data, self.save_embedding_path / "hidden_states" / f"batch_{epoch}.pt")

                sequences_to_decode = [
                    sequence[tokenized_prompt_batch.input_ids.shape[1]:] for sequence in output["sequences"]
                ]

                # Save the outputs until the last hidden state is saved
                output_text.update(
                    {
                        text: tokenizer.decode(sequence, skip_special_tokens=True)
                        for text, sequence in zip(batch, sequences_to_decode)
                    }
                )
                        
                # Explicitly delete tensors and clear cache
                del tokenized_without_prompt, batch_with_prompt, tokenized_batch_with_chat_template, tokenized_prompt_batch, output, sequences_to_decode
                torch.cuda.empty_cache()
                gc.collect()

                epoch += 1

            json.dump(
                output_text, open(self.save_embedding_path / "output_text.json", "w")
            )
